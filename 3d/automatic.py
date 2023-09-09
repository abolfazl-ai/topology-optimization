import time

import numpy as np
import pandas as pd
from cvxopt import cholmod, matrix, spmatrix
from scipy.ndimage import correlate
from scipy.sparse import csc_matrix


def automatic_run(n, ft, filter_bc, r_min, f_name, input_path='input.xlsx'):
    nx, ny, nz = n, n, n
    _, _, _, vf, cf, penalty, _, _, max_it, _, eta, beta, move = read_options(input_path)
    penalCnt, betaCnt = [50, 3, 25, 0.25], [250, 16, 25, 2]
    #   ________________________________________________________________
    node_numbers = np.reshape(range((1 + nx) * (1 + ny) * (1 + nz)), (1 + ny, 1 + nz, 1 + nx), order='F')
    elem_num, dof = nx * ny * nz, (1 + ny) * (1 + nx) * (1 + nz) * 3
    pres, mask = read_pres(input_path, nx, ny, nz)
    free, force = read_bc(input_path, nx, ny, nz, node_numbers)
    densities, elasticities, costs, names, colors = read_materials(input_path)
    Ke, Ke0, c_mat, indexes = element_stiffness(nx, ny, nz, 0.3, node_numbers)
    h, Hs, dHs = prepare_filter(ny, nx, nz, r_min, filter_bc)
    #   ________________________________________________________________
    x, dE, compliance, volume, cost = pres.copy(), np.zeros((ny, nz, nx)), [], [], []
    x[mask] = (vf * (elem_num - pres[~mask].size)) / pres[mask].size
    x_phys, change, loop = x.copy(), 1, 0
    #   ________________________________________________________________
    start = time.time()
    print(f'Starting loop. N={nx}, Filter={ft}, FilterBC={filter_bc.upper()}, Rmin={r_min}')
    while change > 0.0001 and loop < max_it:
        loop += 1
        x_tilde = correlate(x, h, mode=filter_bc) / Hs
        x_phys[mask] = x_tilde[mask]
        if ft > 1:
            f = (np.mean(prj(x_phys, eta, beta)) - vf) * (ft == 3)
            while abs(f) > 1e-6:
                eta = eta - f / np.mean(deta(x_phys.flatten(), eta, beta))
                f = np.mean(prj(x_phys, eta, beta)) - vf
            dHs = Hs / np.reshape(dprj(x_tilde, eta, beta), (ny, nz, nx))
            x_phys = prj(x_phys, eta, beta)
        #   ________________________________________________________________
        P, dP = ordered_simp_interpolation(x_phys, 1 / penalty, densities, costs)
        E, dE = ordered_simp_interpolation(x_phys, penalty, densities, elasticities, True)
        Ks = ((Ke[np.newaxis]).T * E).flatten(order='F')
        K = csc_matrix((Ks, (indexes[:, 0], indexes[:, 1])), shape=(dof, dof))[free, :][:, free].tocoo()
        u, b = np.zeros(dof), force[free]
        K = spmatrix(K.data, K.row.astype(np.int32), K.col.astype(np.int32))
        cholmod.linsolve(K, b)
        u[free] = np.array(b)[:, 0]
        #   ________________________________________________________________
        compliance.append(np.sum(E * np.sum((u[c_mat] @ Ke0) * u[c_mat], axis=1)) / (nx * ny * nz))
        dC = np.reshape(-dE * np.sum((u[c_mat] @ Ke0) * u[c_mat], axis=1), (ny, nz, nx), order='F')
        # dC = correlate(dC / dHs, h, mode=filter_bc)[mask]
        # dP = correlate(dP / dHs, h, mode=filter_bc)[mask]
        #   ________________________________________________________________
        x[mask], vo, co = optimality_criterion(x[mask], dC[mask], P[mask], dP[mask], vf, cf, max(0.15 * 0.96 ** loop, move))
        volume.append(vo)
        cost.append(co)
        change = 1 if loop < 2 else abs((compliance[-2] - compliance[-1]) / compliance[0])
        penalty, beta = cnt(penalty, penalCnt, loop), cnt(beta, betaCnt, loop)
        #   ________________________________________________________________
        print(f"Design cycle {str(loop).rjust(3, '0')}: Change={change:0.6f}, "
              f"Compliance={compliance[-1]:0.3e}, Volume={volume[-1]:0.3f}, Cost={cost[-1]:0.3f}")
        np.save(f_name, np.moveaxis(x_phys, -1, 0))

    print(f'Model converged in {(time.time() - start):0.2f} seconds. Final compliance = {compliance[-1]}')
    return np.moveaxis(x_phys, -1, 0), compliance, volume, cost


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


def ordered_simp_interpolation(x, penalty, X, Y, flatten=False):
    y, dy = np.ones(x.shape), np.zeros(x.shape)
    for i in range(len(X) - 1):
        mask = ((X[i] < x) if i > 0 else True) & (x < X[i + 1] if i < len(X) - 2 else True)
        A = (Y[i] - Y[i + 1]) / (X[i] ** penalty - X[i + 1] ** penalty)
        B = Y[i] - A * (X[i] ** penalty)
        y[mask] = A * (x[mask] ** penalty) + B
        dy[mask] = A * penalty * (x[mask] ** (penalty - 1))
    return y.flatten(order='F') if flatten else y, dy.flatten(order='F') if flatten else dy


def optimality_criterion(x, dC, P, dP, vf, cf, move):
    x_new, xU, xL = x.copy(), x + move, x - move
    LV = [0, 2 * np.max(-dC)]
    LP = [0, 2 * np.max(-dC / (P + x * dP))]
    volume, cost = 0, 0
    while abs(LV[1] - LV[0]) / max(LV[1] + LV[0], 1E-9) > 1e-6 or abs(LP[1] - LP[0]) / max(LP[1] + LP[0], 1E-9) > 1e-6:
        l_mid_v, l_mid_p = 0.5 * (LV[0] + LV[1]), 0.5 * (LP[0] + LP[1])
        B = -dC / (l_mid_v + l_mid_p * P + l_mid_p * x * dP)
        x_new = np.maximum(np.maximum(np.minimum(np.minimum(x * np.sqrt(np.abs(B)), xU), 1), xL), 1E-5)
        volume, cost = np.sum(x_new) / x.size, np.sum(x_new * P) / x.size
        LV[0], LV[1] = (l_mid_v, LV[1]) if volume > vf else (LV[0], l_mid_v)
        LP[0], LP[1] = (l_mid_p, LP[1]) if cost > cf else (LP[0], l_mid_p)
    return x_new, volume, cost


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


def read_options(input_path):
    options = pd.read_excel(input_path, sheet_name='Options')
    nx, ny, nz, vf, cf, penalty, max_it, move, ft, filter_bc, r_min, eta, beta = options['Value']
    nx, ny, nz, penalty, ft, filter_bc = np.array((nx, ny, nz, penalty, ft, filter_bc), dtype=np.int32)
    filter_bc = ['constant', 'reflect', 'nearest', 'mirror', 'wrap'][filter_bc]
    return nx, ny, nz, vf, cf, penalty, ft, filter_bc, max_it, r_min, eta, beta, move


def read_materials(input_path):
    material_df = pd.read_excel(input_path, sheet_name='Materials')
    D = material_df['Density'].tolist()
    E = material_df['Elasticity'].tolist()
    P = material_df['Cost'].tolist()
    E[E == 0] = 1e-4
    P[P == 0] = 1e-4
    M_name = material_df['Material'].tolist()
    M_color = material_df['Color'].tolist()
    return D, E, P, M_name, M_color


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
    pres, mask = np.zeros((ny, nz, nx)), np.ones((ny, nz, nx), dtype=bool)
    for _, row in preserved.iterrows():
        start, end = ([float(x) for x in row['StartPosition'].split(',')],
                      [float(x) for x in row['EndPosition'].split(',')])
        nr = [(int(max(min(np.floor(s * n), np.floor(e * n) - 1), 0)), int(np.floor(e * n)) + 1)
              for n, s, e in list(zip((nx, ny, nz), start, end))]
        mask[nr[1][0]:nr[1][1], nr[2][0]:nr[2][1], nr[0][0]:nr[0][1]] = False
        pres[nr[1][0]:nr[1][1], nr[2][0]:nr[2][1], nr[0][0]:nr[0][1]] = row['Density']
    return pres, mask


arguments = [
    (40, 3, 'reflect', 1.74),
    # (40, 3, 'reflect', 1.80),
    # (40, 3, 'reflect', 2.00),
    # (40, 3, 'reflect', 3.00),
    # (40, 3, 'reflect', 5.00),
    # (40, 3, 'constant', np.sqrt(3)),
    # (40, 3, 'constant', 1.80),
    # (40, 3, 'constant', 2.00),
    # (40, 3, 'constant', 3.00),
    # (40, 3, 'constant', 5.00),
    # (40, 3, 'nearest', np.sqrt(3)),
    # (40, 3, 'mirror', np.sqrt(3)),
    # (40, 3, 'wrap', np.sqrt(3)),
    # (40, 1, 'reflect', np.sqrt(3)),
    # (40, 2, 'reflect', np.sqrt(3)),
]

for nn, fil, fil_bc, r in arguments:
    name_format = f"{nn}-{fil_bc.upper()[0]}-{r:0.2f}-{fil}"
    _, comp, vol_f, pri_f = automatic_run(nn, fil, fil_bc, r, 'runs/' + name_format)

    empty = np.zeros(shape=(500,)) * np.nan
    sheets = ['Compliance', 'Volume', 'Cost']
    dfs = [pd.read_excel('runs/40/data.xlsx', sheet_name=s) for s in sheets]
    for df, sheet_name in zip(dfs, sheets):
        empty[0:len(comp)] = [comp, vol_f, pri_f][sheets.index(sheet_name)]
        df[name_format] = empty
    with pd.ExcelWriter('runs/data.xlsx', engine='openpyxl') as writer:
        [df.to_excel(writer, sheet_name=s, index=False) for df, s in zip(dfs, sheets)]
