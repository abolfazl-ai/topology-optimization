import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib.lines import Line2D
import pandas as pd
from scipy.ndimage import correlate
from scipy.sparse import csc_matrix
from cvxopt import cholmod, matrix, spmatrix
import time


def top2d_mm(input_path='input.xlsx'):
    nx, ny, vol_f, penal, ft, max_it, r_min, eta, beta, move = read_options(input_path)
    penalCnt, betaCnt = [1, 1, 25, 0.25], [1, 1, 25, 2]
    #   ________________________________________________________________
    node_numbers = np.reshape(range((1 + nx) * (1 + ny)), (1 + ny, 1 + nx), order='F')
    nEl, nDof = nx * ny, (1 + ny) * (1 + nx) * 2
    D, E, P, nu, M_name, M_color = read_materials(input_path)

    pres, mask = read_pres(input_path, nx, ny)
    free, F = read_bc(input_path, nx, ny, node_numbers)
    Ke, Ke0, cMat, Iar = element_stiffness(nx, ny, 0.3, node_numbers)

    h, Hs, dHs = prepare_filter(ny, nx, r_min)
    #   ________________________________________________________________
    x, dE_, dV = pres.copy(), np.zeros((ny, nx)), np.zeros((ny, nx))
    dV[mask] = 1 / (nEl * vol_f)
    x[mask] = (vol_f * (nEl - pres[~mask].size)) / pres[mask].size
    xPhys, xOld, ch, loop, U = x.copy(), 1, 1, 0, np.zeros((nDof, 1))
    #   ________________________________________________________________
    fig, ax, im, bg = init_fig(np.reshape(xPhys, (ny, nx), 'F'), D, M_color, M_name)
    start = time.time()
    while ch > 1e-3 and loop < max_it:
        loop += 1
        xTilde = correlate(x, h, mode='reflect') / Hs
        xPhys[mask] = xTilde[mask]
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
        P_, dP_ = ordered_simp_interpolation(xPhys, penal, D, P)
        sK = ((Ke[np.newaxis]).T * E_).flatten(order='F')
        K = csc_matrix((sK, (Iar[:, 0], Iar[:, 1])), shape=(nDof, nDof))[free, :][:, free].tocoo()
        U, B = np.zeros(nDof), F[free]
        K = spmatrix(K.data, K.row.astype(np.int32), K.col.astype(np.int32))
        cholmod.linsolve(K, B)
        U[free] = np.array(B)[:, 0]
        #   ________________________________________________________________
        dc = -dE_ * np.sum((U[cMat] @ Ke0) * U[cMat], axis=1)
        dc = correlate(np.reshape(dc, (ny, nx), order='F') / dHs, h, mode='reflect')
        dV0 = correlate(np.reshape(dV, (ny, nx), 'F') / dHs, h, mode='reflect')
        #   ________________________________________________________________
        # x = oc(nx, ny, x, vol_f, 0.3, dc, P_, dP_, move)
        # x = optimality_criterion2(nx, ny, x, act, vol_f, 0.3, dc, dV0, P_, dP_, move)
        x[mask] = optimality_criterion(x[mask], vol_f, move, dc[mask], dV0[mask])
        penal, beta = cnt(penal, penalCnt, loop), cnt(beta, betaCnt, loop)
        #   ________________________________________________________________
        print(f'Iteration = {loop}, Change = {ch:0.5f}')
        plot(xPhys, loop, ch, fig, ax, im, bg)

    print(f'Model converged in {(time.time() - start):0.2f} seconds')
    ax.set_title(F'Iteration: {loop} | Model converged in {(time.time() - start):0.1f} seconds')
    plt.show(block=True)


def simp_interpolation(x, penal, Y_min, Y_max):
    y = Y_min + x ** penal * (Y_max - Y_min)
    dy = penal * (Y_max - Y_min) * x ** (penal - 1)
    return y.flatten(order='F'), dy.flatten(order='F')


def ordered_simp_interpolation(x, penal, X, Y):
    y, dy = np.ones(x.shape), np.zeros(x.shape)
    for i in range(len(X) - 1):
        mask = ((X[i] < x) if i > 0 else True) & (x < X[i + 1] if i < len(X) - 2 else True)
        A = (Y[i] - Y[i + 1]) / (X[i] ** penal - X[i + 1] ** penal)
        B = Y[i] - A * (X[i] ** penal)
        y[mask] = A * (x[mask] ** penal) + B
        dy[mask] = A * penal * (x[mask] ** (penal - 1))
    return y.flatten(order='F'), dy.flatten(order='F')


def optimality_criterion(x, vol_f, move, dc, dV0):
    x_new, xT = x.copy(), x.copy()
    xU, xL = xT + move, xT - move
    ocP = xT * np.real(np.sqrt(-dc / dV0))
    LM = [0, np.mean(ocP) / vol_f]
    while abs((LM[1] - LM[0]) / (LM[1] + LM[0])) > 1e-4:
        l_mid = 0.5 * (LM[0] + LM[1])
        x_new = np.maximum(np.minimum(np.minimum(ocP / l_mid, xU), 1), xL)
        LM[0], LM[1] = (l_mid, LM[1]) if np.mean(x_new) > vol_f else (LM[0], l_mid)
    return x_new


def optimality_criterion2(nx, ny, x, act, vol_f, cost_f, dc, dV0, P_, dP_, move):
    x_new, xT = x.copy(), x[act]
    xU, xL = xT + move, xT - move
    ocP = xT * np.real(np.sqrt(-dc[act] / dV0[act]))
    LV = [0, np.max(-dc) / vol_f]
    LP = [0, np.max(-dc / (P_ + x * dP_)) / vol_f]
    while abs((LV[1] - LV[0]) / (LV[1] + LV[0])) > 1e-4 or abs((LP[1] - LP[0]) / (LP[1] + LP[0])) > 1e-4:
        l_mid_v, l_mid_p = 0.5 * (LV[0] + LV[1]), 0.5 * (LP[0] + LP[1])
        x_new[act] = np.maximum(np.maximum(np.minimum(np.minimum(
            ocP / (l_mid_v + l_mid_p * P_ + l_mid_p * x * dP_), xU), 1), xL), 1E-5)
        LV[0], LV[1] = (l_mid_v, LV[1]) if np.mean(x_new) > vol_f else (LV[0], l_mid_v)
        LP[0], LP[1] = (l_mid_p, LP[1]) if np.sum(x_new * P_) / (nx * ny) > cost_f else (LP[0], l_mid_p)
    return x_new


def oc(nx, ny, x, vol_f, cost_f, dc, P_, dP_, move):
    dc = -1 * dc
    lV1 = 0
    lV2 = 2 * np.max(dc)
    Temp = P_ + x * dP_
    Temp = dc / Temp
    lP1 = 0
    lP2 = 2 * np.max(Temp)

    while ((lV2 - lV1) / (lV1 + lV2) > 1e-6) or ((lP2 - lP1) / (lP1 + lP2) > 1e-6):
        lmidV = 0.5 * (lV2 + lV1)
        lmidP = 0.5 * (lP2 + lP1)
        Temp = lmidV + lmidP * P_ + lmidP * x * dP_
        Coef = dc / Temp
        Coef = np.abs(Coef)
        xnew = np.maximum(10 ** -5, np.maximum(x - move, np.minimum(1., np.minimum(x + move, x * np.sqrt(Coef)))))

        if np.sum(xnew) - vol_f * nx * ny > 0:
            lV1 = lmidV
        else:
            lV2 = lmidV

        CurrentCostFrac = np.sum(xnew * P_) / (nx * ny)

        if CurrentCostFrac - cost_f > 0:
            lP1 = lmidP
        else:
            lP2 = lmidP

    return xnew


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


def element_stiffness(nx, ny, nu, node_numbers):
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


def init_fig(x, D, colors, names):
    plt.ion()
    fig, ax = plt.subplots()
    cmap = mc.LinearSegmentedColormap.from_list('mesh', list(zip(D, colors)))
    custom_lines = [Line2D([0], [0], marker='o', label='Scatter',
                           lw=0, markerfacecolor=c, markersize=10) for c in colors]
    im = ax.imshow(x, origin='lower', cmap=cmap, vmin=0, vmax=1)
    ax.set_title(F'Iteration: {0}, Change: {1:0.4f}')
    ax.legend(custom_lines, names, ncol=len(colors))  # , bbox_to_anchor=(0.8, -0.15)
    plt.get_current_fig_manager().window.showMaximized()
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


def read_bc(input_path, nx, ny, node_numbers):
    bc = pd.read_excel(input_path, sheet_name='BC')
    fixed, dof = [], 2 * (1 + nx) * (1 + ny)
    force_vector = matrix(0.0, (dof, 1))
    for _, row in bc.iterrows():
        start, end = ([float(x) for x in row['StartPosition'].split(',')],
                      [float(x) for x in row['EndPosition'].split(',')])
        displacement, force = [row['DisX'], row['DisY']], [row['ForceX'], row['ForceY']]
        nr = [(int(np.floor(s * n)), int(np.floor(e * n)) + 1) for n, s, e in list(zip((nx, ny), start, end))]
        nodes = node_numbers[nr[1][0]:nr[1][1], nr[0][0]:nr[0][1]].flatten()
        for i, (d, f) in enumerate(zip(displacement, force)):
            force_vector[(2 * nodes + i).tolist(), 0] = matrix(0 if np.isnan(f) else f, (nodes.size, 1))
            fixed.extend([] if np.isnan(d) else (2 * nodes + i).tolist())
    free = np.setdiff1d(np.arange(0, dof), fixed).tolist()
    return free, force_vector


def read_pres(input_path, nx, ny):
    preserved = pd.read_excel(input_path, sheet_name='Preserved')
    pres, mask = np.zeros((ny, nx)), np.ones((ny, nx), dtype=bool)
    for _, row in preserved.iterrows():
        start, end = ([float(x) for x in row['StartPosition'].split(',')],
                      [float(x) for x in row['EndPosition'].split(',')])
        nr = [(int(max(min(np.floor(s * n), np.floor(e * n) - 1), 0)), int(np.floor(e * n)) + 1)
              for n, s, e in list(zip((nx, ny), start, end))]
        mask[nr[1][0]:nr[1][1], nr[0][0]:nr[0][1]] = False
        pres[nr[1][0]:nr[1][1], nr[0][0]:nr[0][1]] = row['Density']
    return pres, mask


top2d_mm()
