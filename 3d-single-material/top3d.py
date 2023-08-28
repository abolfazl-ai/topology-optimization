import time
import numpy as np
import pandas as pd
from cvxopt import cholmod, matrix, spmatrix
from scipy.ndimage import correlate
from scipy.sparse import csc_matrix, dok_matrix
from plot import plot_3d


def top3d(input_path='input.xlsx'):
    nx, ny, nz, vol_f, penal, ft, max_it, r_min, eta, beta, move = read_options(input_path)
    E_min, E_max, nu = 1E-9, 1.0, 0.3
    penalCnt, betaCnt = [1, 1, 25, 0.25], [1, 1, 25, 2]
    #   ________________________________________________________________
    node_numbers = np.reshape(range((1 + nx) * (1 + ny) * (1 + nz)), (1 + ny, 1 + nz, 1 + nx), order='F')
    nEl, nDof = nx * ny * nz, (1 + ny) * (1 + nx) * (1 + nz) * 3
    pasS, pasV, act = read_pres(input_path, nx, ny, nz)
    free, F = read_bc(input_path, nx, ny, nz, node_numbers)
    Ke, Ke0, cMat, Iar = element_stiffness(nx, ny, nz, nu, node_numbers)
    h, Hs, dHs = prepare_filter(ny, nx, nz, r_min)
    #   ________________________________________________________________
    x, dE_, dV = np.zeros(nEl), np.zeros(nEl), np.zeros(nEl)
    dV[act] = 1 / (nEl * vol_f)
    x[act] = (vol_f * (nEl - len(pasV)) - len(pasS)) / len(act)
    x[pasS] = 1
    xPhys, xOld, ch, loop, U = x.copy(), 1, 1, 0, np.zeros((nDof, 1))
    #   ________________________________________________________________
    start = time.time()
    while ch > 1e-4 and loop < max_it:
        loop += 1
        xTilde = correlate(np.reshape(x, (ny, nz, nx), 'F'), h, mode='reflect') / Hs
        xPhys[act] = xTilde.flatten(order='F')[act]
        if ft > 1:
            f = (np.mean(prj(xPhys, eta, beta)) - vol_f) * (ft == 3)
            while abs(f) > 1e-6:
                eta = eta - f / np.mean(deta(xPhys.flatten(), eta, beta))
                f = np.mean(prj(xPhys, eta, beta)) - vol_f
            dHs = Hs / np.reshape(dprj(xTilde, eta, beta), (ny, nz, nx))
            xPhys = prj(xPhys, eta, beta)
        ch = np.linalg.norm(xPhys - xOld) / np.sqrt(nEl)
        xOld = xPhys.copy()
        #   ________________________________________________________________
        E_, dE_ = simp_interpolation(xPhys, penal, E_min, E_max)
        sK = ((Ke[np.newaxis]).T * E_).flatten(order='F')
        K = csc_matrix((sK, (Iar[:, 0], Iar[:, 1])), shape=(nDof, nDof))[free, :][:, free].tocoo()
        U, B = np.zeros(nDof), F[free]
        K = spmatrix(K.data, K.row.astype(np.int32), K.col.astype(np.int32))
        cholmod.linsolve(K, B)
        U[free] = np.array(B)[:, 0]
        #   ________________________________________________________________
        dc = -dE_ * np.sum((U[cMat] @ Ke0) * U[cMat], axis=1)
        dc = correlate(np.reshape(dc, (ny, nz, nx), order='F') / dHs, h, mode='reflect').flatten(order='F')
        dV0 = correlate(np.reshape(dV, (ny, nz, nx), 'F') / dHs, h, mode='reflect').flatten(order='F')
        #   ________________________________________________________________
        x = optimality_criterion(x, vol_f, act, move, dc, dV0)
        penal, beta = cnt(penal, penalCnt, loop), cnt(beta, betaCnt, loop)
        #   ________________________________________________________________
        print(f"Iteration = {str(loop).rjust(3, '0')}, Change = {ch:0.6f}")
        np.save('x.npy', np.moveaxis(np.reshape(xPhys, (ny, nz, nx), 'F'), -1, 0))

    print(f'Model converged in {(time.time() - start):0.2f} seconds')
    np.save('final.npy', np.moveaxis(np.reshape(xPhys, (ny, nz, nx), 'F'), -1, 0))
    plot_3d('final.npy', [1, ], ['Solid', ], ['gray', ])


def prj(v, eta, beta):
    return (np.tanh(beta * eta) + np.tanh(beta * (v - eta))) / (np.tanh(beta * eta) + np.tanh(beta * (1 - eta)))


def deta(v, eta, beta):
    return (-beta * (1 / np.sinh(beta)) * ((1 / np.cosh(beta * (v - eta))) ** 2) *
            (np.sinh(v * beta)) * (np.sinh((1 - v) * beta)))


def dprj(v, eta, beta):
    return beta * (1 - np.tanh(beta * (v - eta)) ** 2) / (np.tanh(beta * eta) + np.tanh(beta * (1 - eta)))


def cnt(v, vCnt, el):
    condition = (el >= vCnt[0]) and (v < vCnt[1]) and (el % vCnt[2] == 0)
    return v + condition * vCnt[3]


def simp_interpolation(x, penal, Y_min, Y_max):
    y = Y_min + x ** penal * (Y_max - Y_min)
    dy = penal * (Y_max - Y_min) * x ** (penal - 1)
    return y, dy


def prepare_filter(ny, nx, nz, r_min):
    dy, dz, dx = np.meshgrid(np.arange(-np.ceil(r_min) + 1, np.ceil(r_min)),
                             np.arange(-np.ceil(r_min) + 1, np.ceil(r_min)),
                             np.arange(-np.ceil(r_min) + 1, np.ceil(r_min)))
    h = np.maximum(0, r_min - np.sqrt(dx ** 2 + dy ** 2 + dz ** 2))
    Hs = correlate(np.ones((ny, nz, nx)), h, mode='reflect')
    return h, Hs, Hs.copy()


def optimality_criterion(x, vol_f, act, move, dc, dV0):
    x_new, xT = x.copy(), x[act]
    xU, xL = xT + move, xT - move
    ocP = xT * np.real(np.sqrt(-dc[act] / dV0[act]))
    LM = [0, np.mean(ocP) / vol_f]
    while abs((LM[1] - LM[0]) / (LM[1] + LM[0])) > 1e-4:
        l_mid = 0.5 * (LM[0] + LM[1])
        x_new[act] = np.maximum(np.minimum(np.minimum(ocP / l_mid, xU), 1), xL)
        LM[0], LM[1] = (l_mid, LM[1]) if np.mean(x_new) > vol_f else (LM[0], l_mid)
    return x_new


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


def read_options(input_path):
    options = pd.read_excel(input_path, sheet_name='Options')
    nx, ny, nz, vol_f, penal, ft, max_it, r_min, eta, beta, move = options['Value']
    nx, ny, nz, penal, ft, max_it = np.array((nx, ny, nz, penal, ft, max_it), dtype=np.int32)
    return nx, ny, nz, vol_f, penal, ft, max_it, r_min, eta, beta, move


def read_bc(input_path, nx, ny, nz, node_numbers):
    bc = pd.read_excel(input_path, sheet_name='BC')
    fixed, dof = [], 3 * (1 + nx) * (1 + ny) * (1 + nz)
    force_vector = matrix(0.0, (dof, 1))
    for _, row in bc.iterrows():
        start, end = ([float(x) for x in row['StartPosition'].split(',')],
                      [float(x) for x in row['EndPosition'].split(',')])
        displacement, force = [row['DisX'], row['DisY'], row['DisZ']], [row['ForceX'], row['ForceY'], row['ForceZ']]
        nr = [(int(np.floor(s * n)), int(np.floor(e * n)) + 1) for n, s, e in list(zip((nx, ny, nz), start, end))]
        nodes = node_numbers[nr[1][0]:nr[1][1], nr[2][0]:nr[2][1], nr[0][0]:nr[0][1]].flatten()
        for i, (d, f) in enumerate(zip(displacement, force)):
            force_vector[(3 * nodes + i).tolist(), 0] = matrix(0 if np.isnan(f) else f, (nodes.size, 1))
            fixed.extend([] if np.isnan(d) else (3 * nodes + i).tolist())
    free = np.setdiff1d(np.arange(0, dof), fixed).tolist()
    return free, force_vector


def read_pres(input_path, nx, ny, nz):
    preserved = pd.read_excel(input_path, sheet_name='PreservedVolume')
    pasS, pasV = [], []
    for _, row in preserved.iterrows():
        left, right, top, bottom = row['Left'], row['Right'], row['Top'], row['Bottom']
        elements = [range(int(x * nx * ny + (1 - top) * ny), int(x * nx * ny + (1 - bottom) * ny))
                    for x in np.arange(left, right, step=1 / nx)]
        pasS.extend(elements) if row['Density'] == 1 else pasV.extend(elements)
    act = np.setdiff1d(np.arange(0, nx * ny * nz), np.union1d(np.array(pasS), np.array(pasV))).tolist()
    return pasS, pasV, act


top3d()
