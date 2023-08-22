import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import correlate
from scipy.sparse import csc_matrix
from cvxopt import cholmod, matrix, spmatrix
import time


def top2d_mm(input_path='input.xlsx'):
    nx, ny, vol_f, penal, ft, max_it, r_min, eta, beta, move = read_options(input_path)
    penalCnt, betaCnt = [1, 1, 25, 0.25], [1, 1, 25, 2]
    #   ________________________________________________________________
    nEl, nDof = nx * ny, (1 + ny) * (1 + nx) * 2
    D, E, P, nu, M_name, M_color = read_materials(input_path)
    free, F = read_bc(input_path, nx, ny, nDof)
    pasS, pasV, act = read_pres(input_path, nx, ny)
    Ke, Ke0, cMat, Iar = element_stiffness(nx, ny, 0.3)
    h, Hs, dHs = prepare_filter(ny, nx, r_min)
    #   ________________________________________________________________
    x, dE_, dV = np.zeros(nEl), np.zeros(nEl), np.zeros(nEl)
    dV[act] = 1 / (nEl * vol_f)
    x[act] = (vol_f * (nEl - len(pasV)) - len(pasS)) / len(act)
    x[pasS] = 1
    xPhys, xOld, ch, loop, U = x.copy(), 1, 1, 0, np.zeros((nDof, 1))
    #   ________________________________________________________________
    fig, ax, im, bg = init_fig(1 - np.reshape(xPhys, (ny, nx), 'F'))
    start = time.time()
    while ch > 1e-4 and loop < max_it:
        loop += 1
        xTilde = correlate(np.reshape(x, (ny, nx), 'F'), h, mode='reflect') / Hs
        xPhys[act] = xTilde.flatten(order='F')[act]
        if ft > 1:
            f = (np.mean(prj(xPhys, eta, beta)) - vol_f) * (ft == 3)
            while abs(f) > 1e-6:
                eta = eta - f / np.mean(deta(xPhys.flatten(), eta, beta))
                f = np.mean(prj(xPhys, eta, beta)) - vol_f
            dHs = Hs / np.reshape(dprj(xTilde, eta, beta), (ny, nx))
            xPhys = prj(xPhys, eta, beta)
        ch = np.linalg.norm(xPhys - xOld) / np.sqrt(nEl)
        xOld = xPhys.copy()
        #   ________________________________________________________________
        E_, dE_ = ordered_simp_interpolation(xPhys, penal, D, E)
        P_, dP_ = ordered_simp_interpolation(xPhys, 1 / penal, D, P)
        sK = ((Ke[np.newaxis]).T * E_).flatten(order='F')
        K = csc_matrix((sK, (Iar[:, 0], Iar[:, 1])), shape=(nDof, nDof))[free, :][:, free].tocoo()
        U, B = np.zeros(nDof), F[free, 0]
        K = spmatrix(K.data, K.row.astype(np.int32), K.col.astype(np.int32))
        cholmod.linsolve(K, B)
        U[free] = np.array(B)[:, 0]
        #   ________________________________________________________________
        dc = -dE_ * np.sum((U[cMat] @ Ke0) * U[cMat], axis=1)
        dc = correlate(np.reshape(dc, (ny, nx), order='F') / dHs, h, mode='reflect').flatten(order='F')
        dV0 = correlate(np.reshape(dV, (ny, nx), 'F') / dHs, h, mode='reflect').flatten(order='F')
        #   ________________________________________________________________
        x = optimality_criterion(x, vol_f, act, move, dc, dV0)
        penal, beta = cnt(penal, penalCnt, loop), cnt(beta, betaCnt, loop)
        #   ________________________________________________________________
        print(f'Iteration = {loop}, Change = {ch:0.5f}')
        plot(1 - np.reshape(xPhys, (ny, nx), 'F'), loop, ch, fig, ax, im, bg)

    print(f'Model converged in {(time.time() - start):0.2f} seconds')
    ax.set_title(F'Iteration: {loop} | Model converged in {(time.time() - start):0.1f} seconds')
    plt.show(block=True)


def oc(n_elx, n_ely, vol_frac, x, cost_frac, dc, P_, dP_, loop, MinMove):
    Temp = -dc / (P_ + x * dP_)
    lV1, lV2, lP1, lP2 = 0.0, 2 * np.max(-dc), 0.0, 2 * np.max(Temp)
    move = max(0.15 * 0.96 ** loop, MinMove)
    x_new = np.zeros(x.shape)
    while ((lV2 - lV1) / (lV1 + lV2) > 1e-6) or ((lP2 - lP1) / (lP1 + lP2) > 1e-6):
        lmidV = 0.5 * (lV2 + lV1)
        lmidP = 0.5 * (lP2 + lP1)
        Temp = lmidV + lmidP * P_ + lmidP * x * dP_
        Coef = -dc / Temp
        Coef = np.abs(Coef)
        x_new = np.maximum(10 ** -5, np.maximum(x - move, np.minimum(1., np.minimum(x + move, x * np.sqrt(Coef)))))
        if np.sum(x_new) - vol_frac * n_elx * n_ely > 0:
            lV1 = lmidV
        else:
            lV2 = lmidV

        CurrentCostFrac = np.sum(x_new * P_) / (n_elx * n_ely)
        if CurrentCostFrac - cost_frac > 0:
            lP1 = lmidP
        else:
            lP2 = lmidP

    return x_new


def ordered_simp_interpolation(x, penal, X, Y):
    y, dy = np.zeros(x.shape), np.zeros(x.shape)
    for i, xi in enumerate(x):
        for j in range(len(X)):
            if (X[j] < xi) and (X[j + 1] >= xi):
                A = (Y[j] - Y[j + 1]) / (X[j] ** penal - X[j + 1] ** penal)
                B = Y[j] - A * (X[j] ** penal)
                y[i] = max(A * (xi ** penal) + B, min(Y))
                dy[i] = A * penal * (xi ** (penal - 1))

    return y, dy


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


def prepare_filter(ny, nx, r_min):
    dy, dx = np.meshgrid(np.arange(-np.ceil(r_min) + 1, np.ceil(r_min)),
                         np.arange(-np.ceil(r_min) + 1, np.ceil(r_min)))
    h = np.maximum(0, r_min - np.sqrt(dx ** 2 + dy ** 2))
    Hs = correlate(np.ones((ny, nx)), h, mode='reflect')
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


def element_stiffness(nx, ny, nu):
    node_numbers = np.reshape(range((1 + nx) * (1 + ny)), (1 + ny, 1 + nx), order='F')
    cVec = np.reshape(2 * node_numbers[0: -1, 0: -1] + 2, (nx * ny, 1), order='F')
    cMat = cVec + np.array([0, 1, 2 * ny + 2, 2 * ny + 3, 2 * ny + 0, 2 * ny + 1, -2, -1])
    sI, sII = np.hstack([np.arange(j, 8) for j in range(8)]), np.hstack([np.tile(j, 7 - j + 1) for j in range(8)])
    iK, jK = cMat[:, sI].T, cMat[:, sII].T
    Iar = np.sort(np.hstack((iK.reshape((-1, 1), order='F'), jK.reshape((-1, 1), order='F'))))[:, [1, 0]]
    c1 = np.array([12, 3, -6, -3, -6, -3, 0, 3, 12, 3, 0, -3, -6, -3, -6, 12, -3, 0,
                   -3, -6, 3, 12, 3, -6, 3, -6, 12, 3, -6, -3, 12, 3, 0, 12, -3, 12])
    c2 = np.array([-4, 3, -2, 9, 2, -3, 4, -9, -4, -9, 4, -3, 2, 9, -2, -4, -3, 4,
                   9, 2, 3, -4, -9, -2, 3, 2, -4, 3, -2, 9, -4, -9, 4, -4, -3, -4])
    Ke = 1 / (1 - nu ** 2) / 24 * (c1 + nu * c2)
    Ke0 = np.zeros((8, 8))
    Ke0[np.triu_indices(8)] = Ke
    Ke0 = Ke0 + Ke0.T - np.diag(np.diag(Ke0))
    return Ke, Ke0, cMat, Iar


def init_fig(x):
    plt.ion()  # Ensure that redrawing is possible
    fig, ax = plt.subplots()
    im = ax.imshow(x, cmap='gray', vmin=0, vmax=1)
    ax.set_title(F'Iteration: {0}, Change: {1:0.4f}')
    plt.pause(0.1)
    bg = fig.canvas.copy_from_bbox(fig.bbox)
    ax.draw_artist(im)
    fig.canvas.blit(fig.bbox)
    return fig, ax, im, bg


def plot(x, loop, ch, fig, ax, im, bg):
    fig.canvas.restore_region(bg)
    im.set_array(x)
    ax.set_title(F'Iteration: {loop}, Change: {ch:0.5f}')
    ax.draw_artist(im)
    fig.canvas.blit(fig.bbox)
    fig.canvas.flush_events()


def read_materials(input_path):
    material_df = pd.read_excel(input_path, sheet_name='Materials')
    D = material_df['Density'].tolist()
    E = material_df['Elasticity'].tolist()
    P = material_df['Price'].tolist()
    nu = material_df['Poisson'].tolist()
    M_name = material_df['Material'].tolist()
    M_color = material_df['Color'].tolist()
    return D, E, P, nu, M_name, M_color


def read_options(input_path):
    options = pd.read_excel(input_path, sheet_name='Options')
    nx, ny, vol_f, penal, ft, max_it, r_min, eta, beta, move = options['Value']
    nx, ny, penal, ft, max_it = np.array((nx, ny, penal, ft, max_it), dtype=np.int32)
    return nx, ny, vol_f, penal, ft, max_it, r_min, eta, beta, move


def read_bc(input_path, nx, ny, nDof):
    bc = pd.read_excel(input_path, sheet_name='BC')
    nodes = {}
    for _, row in bc.iterrows():
        sx, sy, ex, ey = row['StartX'], row['StartY'], row['EndX'], row['EndY']
        for x in np.linspace(sx, ex, int(max(1, (ex - sx) * (nx + 1)))):
            for y in np.linspace(sy, ey, int(max(1, (ey - sy) * (ny + 1)))):
                node_number = round((ny + 1) * x * nx + (1 - y) * ny)
                node_x, node_y = node_number * 2, node_number * 2 + 1
                nodes[node_x], nodes[node_y] = (row['DisX'], row['ForceX']), (row['DisY'], row['ForceY'])
    fixed = [i for i, (d, _) in nodes.items() if d == 0]
    free = np.setdiff1d(np.arange(0, nDof), fixed).tolist()
    Fd = {i: f for i, (_, f) in nodes.items() if not np.isnan(f)}
    F = csc_matrix((list(Fd.values()), (list(Fd.keys()), np.zeros(len(Fd)))), shape=(nDof, 1))
    return free, matrix(F.toarray())


def read_pres(input_path, nx, ny):
    preserved = pd.read_excel(input_path, sheet_name='Preserved')
    pasS, pasV = [], []
    for _, row in preserved.iterrows():
        left, right, top, bottom = row['Left'], row['Right'], row['Top'], row['Bottom']
        elements = [range(int(x * nx * ny + (1 - top) * ny), int(x * nx * ny + (1 - bottom) * ny))
                    for x in np.arange(left, right, step=1 / nx)]
        pasS.extend(elements) if row['Density'] == 1 else pasV.extend(elements)
    act = np.setdiff1d(np.arange(0, nx * ny), np.union1d(np.array(pasS), np.array(pasV))).tolist()
    return pasS, pasV, act


top2d_mm()
