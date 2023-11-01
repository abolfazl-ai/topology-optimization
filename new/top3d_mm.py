import numpy as np
from cvxopt import cholmod, spmatrix
from scipy.ndimage import correlate
from scipy.sparse import csc_matrix
import time


def top3d_mm(nx, ny, nz, node_numbers, volume_fraction,  # Mesh specifications and volume constraint
             free, force, pres, mask,  # Boundary conditions and preserved regions
             densities, elasticities,  # Materials
             ft, filter_bc, r_min, eta, beta,  # Filter options
             max_it, x_con, c_con, move, penalty, penal_cnt, beta_cnt, move_cnt,  # Optimization options
             iter_callback):
    elem_num, dof = nx * ny * nz, (1 + ny) * (1 + nx) * (1 + nz) * 3
    Ke, Ke0, c_mat, indexes = element_stiffness(nx, ny, nz, 0.3, node_numbers)
    h, Hs, dHs = prepare_filter(ny, nx, nz, r_min, filter_bc)
    #   ________________________________________________________________
    x, dE, dV, compliance, volume = pres.copy(), np.zeros((ny, nz, nx)), np.zeros((ny, nz, nx)), [], []
    dV[mask] = 1 / (elem_num * volume_fraction)
    x[mask] = (volume_fraction * (elem_num - pres[~mask].size)) / pres[mask].size
    x_phys, x_old, c_change, x_change, loop = x.copy(), x.copy(), 1, 1, 0
    while c_change > c_con and x_change > x_con and loop < max_it:
        loop += 1
        x_tilde = correlate(x, h, mode=filter_bc) / Hs
        x_phys[mask] = x_tilde[mask]
        if ft > 1:
            f = (np.mean(prj(x_phys, eta, beta)) - volume_fraction) * (ft == 3)
            while abs(f) > 1e-6:
                eta = eta - f / np.mean(deta(x_phys.flatten(), eta, beta))
                f = np.mean(prj(x_phys, eta, beta)) - volume_fraction
            dHs = Hs / np.reshape(dprj(x_tilde, eta, beta), (ny, nz, nx))
            x_phys[mask] = prj(x_phys, eta, beta)[mask]
        #   ________________________________________________________________
        E, dE = ordered_simp_interpolation(x_phys, penalty, densities, elasticities, True)
        assem_start = time.time()
        Ks = ((Ke[np.newaxis]).T * E).flatten(order='F')
        K = csc_matrix((Ks, (indexes[:, 0], indexes[:, 1])), shape=(dof, dof))[free, :][:, free].tocoo()
        K = spmatrix(K.data, K.row.astype(np.int32), K.col.astype(np.int32))
        assem_end = time.time()
        print(f'Assembled in {assem_end - assem_start} s')

        solve_start = time.time()
        u, b = np.zeros(dof), force[free]
        cholmod.linsolve(K, b)
        u[free] = np.array(b)[:, 0]
        solve_end = time.time()
        print(f'Solved in {solve_end - solve_start} s')
        #   ________________________________________________________________
        dC = np.reshape(-dE * np.sum((u[c_mat] @ Ke0) * u[c_mat], axis=1), (ny, nz, nx), order='F')
        dC = correlate(dC / dHs, h, mode=filter_bc)
        dV0 = correlate(dV / dHs, h, mode=filter_bc)
        x[mask] = optimality_criterion(x, volume_fraction, move, dC, dV0, mask)[mask]
        #   ________________________________________________________________
        volume.append(np.mean(x))
        compliance.append(np.sum(E * np.sum((u[c_mat] @ Ke0) * u[c_mat], axis=1)) / (nx * ny * nz))
        c_change = move if loop < 3 else abs(np.sqrt((compliance[-2] - compliance[-1]) ** 2 +
                                                     (compliance[-3] - compliance[-2]) ** 2) / compliance[0])
        x_change = move if loop < 3 else np.linalg.norm(x_phys - x_old) / np.sqrt(elem_num)
        x_old = x_phys.copy()
        penalty, beta, move = cnt(penalty, penal_cnt, loop), cnt(beta, beta_cnt, loop), cnt(move, move_cnt, loop)
        iter_callback(loop, x_phys, x_change, c_change, compliance[-1], volume[-1])

    return x_phys, compliance, volume


def prj(v, eta, beta):
    return (np.tanh(beta * eta) + np.tanh(beta * (v - eta))) / (np.tanh(beta * eta) + np.tanh(beta * (1 - eta)))


def deta(v, eta, beta):
    return (-beta * (1 / np.sinh(beta)) * ((1 / np.cosh(beta * (v - eta))) ** 2) *
            (np.sinh(v * beta)) * (np.sinh((1 - v) * beta)))


def dprj(v, eta, beta):
    return beta * (1 - np.tanh(beta * (v - eta)) ** 2) / (np.tanh(beta * eta) + np.tanh(beta * (1 - eta)))


def cnt(v, vCnt, el):
    condition = (el >= vCnt[0]) and (abs(v - vCnt[1]) >= abs(vCnt[3])) and (el % vCnt[2] == 0)
    return v + condition * vCnt[3]


def ordered_simp_interpolation(x, penalty, X, Y, flatten=False):
    y, dy = np.ones(x.shape), np.zeros(x.shape)
    for i in range(len(X) - 1):
        mask = ((X[i] < x) if i > 0 else True) & (x < X[i + 1] if i < len(X) - 2 else True)
        A = (Y[i] - Y[i + 1]) / (X[i] ** penalty - X[i + 1] ** penalty)
        B = Y[i] - A * (X[i] ** penalty)
        y[mask] = A * (x[mask] ** penalty) + B
        dy[mask] = A * penalty * (x[mask] ** (penalty - 1))
    return y.flatten(order='F') if flatten else y, dy.flatten(order='F') if flatten else dy


def optimality_criterion(x, vol_f, move, dc, dV0, mask):
    x_new, xT = x.copy(), x[mask]
    xU, xL = xT + move, xT - move
    ocP = xT * np.real(np.sqrt(-dc[mask] / dV0[mask]))
    LM = [0, np.mean(ocP) / vol_f]
    while abs((LM[1] - LM[0]) / (LM[1] + LM[0])) > 1e-4:
        l_mid = 0.5 * (LM[0] + LM[1])
        x_new[mask] = np.maximum(np.minimum(np.minimum(ocP / l_mid, xU), 1), xL)
        LM[0], LM[1] = (l_mid, LM[1]) if np.mean(x_new) > vol_f else (LM[0], l_mid)
    return x_new


def prepare_filter(ny, nx, nz, r_min, filter_bc):
    dy, dz, dx = np.meshgrid(np.arange(-np.ceil(r_min) + 1, np.ceil(r_min)),
                             np.arange(-np.ceil(r_min) + 1, np.ceil(r_min)),
                             np.arange(-np.ceil(r_min) + 1, np.ceil(r_min)))
    h = np.maximum(0, r_min - np.sqrt(dx ** 2 + dy ** 2 + dz ** 2))
    Hs = correlate(np.ones((ny, nz, nx)), h, mode=filter_bc)
    return h, Hs, Hs.copy()


def element_stiffness(nx, ny, nz, nu, node_numbers):
    cVec = np.reshape(3 * node_numbers[0: -1, 0: -1, 0: -1] + 3, (nx * ny * nz, 1), order='F')
    additions = [np.array([0, 1, 2], dtype=np.int32),
                 np.array([3 * (ny + 1) * (nz + 1) + i for i in [0, 1, 2, -3, -2, -1]], dtype=np.int32),
                 np.array([-3, -2, -1], dtype=np.int32),
                 np.array([3 * (ny + 1) + i for i in [0, 1, 2]], dtype=np.int32),
                 np.array([3 * (ny + 1) * (nz + 2) + i for i in [0, 1, 2, -3, -2, -1]], dtype=np.int32),
                 np.array([3 * (ny + 1) + i for i in [-3, -2, -1]], dtype=np.int32)]
    cMat = cVec + np.concatenate(additions)
    sI, sII = np.hstack([np.arange(j, 24) for j in range(24)]), np.hstack([np.tile(j, 24 - j) for j in range(24)])
    iK, jK = cMat[:, sI].T, cMat[:, sII].T
    Iar = np.sort(np.hstack((iK.reshape((-1, 1), order='F'), jK.reshape((-1, 1), order='F'))))[:, [1, 0]]
    Ke = (1 / (1 + nu) / (2 * nu - 1) / 144 *
          (np.array([-32, -6, -6, 8, 6, 6, 10, 6, 3, -4, -6, -3, -4, -3, -6, 10,
                     3, 6, 8, 3, 3, 4, -3, -3, -32, -6, -6, -4, -3, 6, 10, 3, 6, 8, 6, -3, -4, -6, -3, 4, -3, 3, 8, 3,
                     3, 10, 6, -32, -6, -3, -4, -3, -3, 4, -3, -6, -4, 6, 6, 8, 6, 3, 10, 3, 3, 8, 3, 6, 10, -32, 6, 6,
                     -4, 6, 3, 10, -6, -3, 10, -3, -6, -4, 3, 6, 4, 3, 3, 8, -3, -3, -32, -6, -6, 8, 6, -6, 10, 3, 3, 4,
                     -3, 3, -4, -6, -3, 10, 6, -3, 8, 3, -32, 3, -6, -4, 3, -3, 4, -6, 3, 10, -6, 6, 8, -3, 6, 10, -3,
                     3, 8, -32, -6, 6, 8, 6, -6, 8, 3, -3, 4, -3, 3, -4, -3, 6, 10, 3, -6, -32, 6, -6, -4, 3, 3, 8, -3,
                     3, 10, -6, -3, -4, 6, -3, 4, 3, -32, 6, 3, -4, -3, -3, 8, -3, -6, 10, -6, -6, 8, -6, -3, 10, -32,
                     6, -6, 4, 3, -3, 8, -3, 3, 10, -3, 6, -4, 3, -6, -32, 6, -3, 10, -6, -3, 8, -3, 3, 4, 3, 3, -4, 6,
                     -32, 3, -6, 10, 3, -3, 8, 6, -3, 10, 6, -6, 8, -32, -6, 6, 8, 6, -6, 10, 6, -3, -4, -6, 3, -32, 6,
                     -6, -4, 3, 6, 10, -3, 6, 8, -6, -32, 6, 3, -4, 3, 3, 4, 3, 6, -4, -32, 6, -6, -4, 6, -3, 10, -6, 3,
                     -32, 6, -6, 8, -6, -6, 10, -3, -32, -3, 6, -4, -3, 3, 4, -32, -6, -6, 8, 6, 6, -32, -6, -6, -4,
                     -3, -32, -6, -3, -4, -32, 6, 6, -32, -6, -32]) + nu *
           np.array([48, 0, 0, 0, -24, -24, -12, 0, -12, 0, 24, 0, 0, 0, 24, -12, -12, 0, -12, 0, 0, -12, 12,
                     12, 48, 0, 24, 0, 0, 0, -12, -12, -24, 0, -24, 0, 0, 24, 12, -12, 12, 0, -12, 0, -12, -12,
                     0, 48, 24, 0, 0, 12, 12, -12, 0, 24, 0, -24, -24, 0, 0, -12, -12, 0, 0, -12, -12, 0, -12,
                     48, 0, 0, 0, -24, 0, -12, 0, 12, -12, 12, 0, 0, 0, -24, -12, -12, -12, -12, 0, 0, 48, 0,
                     24, 0, -24, 0, -12, -12, -12, -12, 12, 0, 0, 24, 12, -12, 0, 0, -12, 0, 48, 0, 24, 0, -12,
                     12, -12, 0, -12, -12, 24, -24, 0, 12, 0, -12, 0, 0, -12, 48, 0, 0, 0, -24, 24, -12, 0, 0,
                     -12, 12, -12, 0, 0, -24, -12, -12, 0, 48, 0, 24, 0, 0, 0, -12, 0, -12, -12, 0, 0, 0, -24,
                     12, -12, -12, 48, -24, 0, 0, 0, 0, -12, 12, 0, -12, 24, 24, 0, 0, 12, -12, 48, 0, 0, -12,
                     -12, 12, -12, 0, 0, -12, 12, 0, 0, 0, 24, 48, 0, 12, -12, 0, 0, -12, 0, -12, -12, -12, 0,
                     0, -24, 48, -12, 0, -12, 0, 0, -12, 0, 12, -12, -24, 24, 0, 48, 0, 0, 0, -24, 24, -12, 0,
                     12, 0, 24, 0, 48, 0, 24, 0, 0, 0, -12, 12, -24, 0, 24, 48, -24, 0, 0, -12, -12, -12, 0,
                     -24, 0, 48, 0, 0, 0, -24, 0, -12, 0, -12, 48, 0, 24, 0, 24, 0, -12, 12, 48, 0, -24, 0, 12,
                     -12, -12, 48, 0, 0, 0, -24, -24, 48, 0, 24, 0, 0, 48, 24, 0, 0, 48, 0, 0, 48, 0, 48])))
    Ke0 = np.zeros((24, 24))
    Ke0[np.triu_indices(24)] = Ke
    Ke0 = Ke0 + Ke0.T - np.diag(np.diag(Ke0))
    return Ke, Ke0, cMat, Iar
