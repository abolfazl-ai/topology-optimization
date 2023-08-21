import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import correlate
from scipy.sparse import csc_matrix
from cvxopt import cholmod, matrix, spmatrix
import time


def main(input_path='input.xlsx'):
    nx, ny, vol_f, penal, ft, max_it, r_min, eta, beta, move = read_options(input_path)
    E_min, E_max, nu = 1E-9, 1.0, 0.3
    penalCnt, betaCnt = [1, 1, 25, 0.25], [1, 1, 25, 2]
    #   ________________________________________________________________
    nEl, nDof = nx * ny, (1 + ny) * (1 + nx) * 2
    pasS, pasV, _, _, act = bc_load(nx, ny, nDof)
    free, F = read_bc(input_path, nx, ny, nDof)
    Ke, Ke0, cMat, Iar = element_stiffness(nx, ny, nu)
    h, Hs, dHs = prepare_filter(ny, nx, r_min)
    #   ________________________________________________________________
    x, dsK, dV = np.zeros(nEl), np.zeros(nEl), np.zeros(nEl)
    dV[act] = 1 / (nEl * vol_f)
    x[act] = (vol_f * (nEl - len(pasV)) - len(pasS)) / len(act)
    x[pasS] = 1
    xPhys, xOld, ch, loop, U = x.copy(), 1, 1, 0, np.zeros((nDof, 1))
    #   ________________________________________________________________
    fig, ax, im, bg = init_fig(1 - np.reshape(xPhys, (ny, nx), 'F'))
    start = time.time()
    while ch > 1e-5 and loop < max_it:
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
        dsK[act] = -penal * (E_max - E_min) * xPhys[act] ** (penal - 1)
        sK = ((Ke[np.newaxis]).T * (E_min + xPhys ** penal * (E_max - E_min))).flatten(order='F')
        K = csc_matrix((sK, (Iar[:, 0], Iar[:, 1])), shape=(nDof, nDof))[free, :][:, free].tocoo()
        U, B = np.zeros(nDof), F[free, 0]
        K = spmatrix(K.data, K.row.astype(np.int32), K.col.astype(np.int32))
        cholmod.linsolve(K, B)
        U[free] = np.array(B)[:, 0]
        #   ________________________________________________________________
        dc = dsK * np.sum((U[cMat] @ Ke0) * U[cMat], axis=1)
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


def bc_load(nx, ny, nDof):
    fixed = np.union1d(np.arange(0, 2 * (ny + 1), 2), 2 * (1 + nx) * (1 + ny) - 1)
    pasS, pasV = [], []
    F = matrix(csc_matrix(([-1.0], ([1], [0])), shape=(nDof, 1)).toarray())
    free = np.setdiff1d(np.arange(0, nDof), fixed).tolist()
    act = np.setdiff1d(np.arange(0, nx * ny), np.union1d(pasS, pasV)).tolist()
    return pasS, pasV, F, free, act


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


def read_options(input_path):
    options = pd.read_excel(input_path, sheet_name='Options')
    nx, ny, vol_f, penal, ft, max_it, r_min, eta, beta, move = options['Value']
    nx, ny, penal, ft, max_it = np.array((nx, ny, penal, ft, max_it), dtype=np.int32)
    return nx, ny, vol_f, penal, ft, max_it, r_min, eta, beta, move


def read_bc(input_path, nx, ny, nDof):
    bc = pd.read_excel(input_path, sheet_name='BC')
    nodes = {}
    for index, row in bc.iterrows():
        sx, sy, ex, ey = row['StartX'], row['StartY'], row['EndX'], row['EndY']
        if sx == ex and sy == ey:
            node_x, node_y = get_node_dof(sx, sy, nx, ny)
            nodes[node_x], nodes[node_y] = (row['DisX'], row['ForceX']), (row['DisY'], row['ForceY'])
        else:
            for x in np.linspace(sx, ex, int(max(1, (ex - sx) * (nx + 1)))):
                for y in np.linspace(sy, ey, int(max(1, (ey - sy) * (ny + 1)))):
                    node_x, node_y = get_node_dof(x, y, nx, ny)
                    nodes[node_x], nodes[node_y] = (row['DisX'], row['ForceX']), (row['DisY'], row['ForceY'])

    fixed = [i for i, (d, _) in nodes.items() if d == 0]
    free = np.setdiff1d(np.arange(0, nDof), fixed).tolist()
    Fd = {i: f for i, (_, f) in nodes.items() if not np.isnan(f)}
    F = csc_matrix((list(Fd.values()), (list(Fd.keys()), np.zeros(len(Fd), dtype=np.int32))), shape=(nDof, 1))
    return free, matrix(F.toarray())

    # preserved = pd.read_excel(input_path, sheet_name='Preserved')


def get_node_dof(x, y, nx, ny):
    n, m = x * nx, (1 - y) * ny
    node_number = round((ny + 1) * n + m)
    return node_number * 2, node_number * 2 + 1


main()
