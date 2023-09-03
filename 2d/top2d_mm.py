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
    nx, ny, vf, cf, penalty, ft, max_it, r_min, eta, beta, move = read_options(input_path)
    penalCnt, betaCnt = [1, 1, 25, 0.25], [1, 1, 25, 2]
    #   ________________________________________________________________
    node_numbers = np.reshape(range((1 + nx) * (1 + ny)), (1 + ny, 1 + nx), order='F')
    elem_num, dof = nx * ny, (1 + ny) * (1 + nx) * 2
    densities, elasticities, costs, names, colors = read_materials(input_path)
    pres, mask = read_pres(input_path, nx, ny)
    free, force = read_bc(input_path, nx, ny, node_numbers)
    Ke, Ke0, c_mat, indexes = element_stiffness(nx, ny, 0.3, node_numbers)
    h, Hs, dHs = prepare_filter(ny, nx, r_min)
    #   ________________________________________________________________
    x, dE = pres.copy(), np.zeros((ny, nx))
    x[mask] = (vf * (elem_num - pres[~mask].size)) / pres[mask].size
    x_phys, x_old, change, loop = x.copy(), 1, 1, 0
    #   ________________________________________________________________
    fig, ax, im, bg = init_fig(np.reshape(x_phys, (ny, nx), 'F'), densities, colors, names)
    start = time.time()
    while change > 1e-3 and loop < max_it:
        loop += 1
        x_tilde = correlate(x, h, mode='reflect') / Hs
        x_phys[mask] = x_tilde[mask]
        if ft > 1:
            f = (np.mean(prj(x_phys, eta, beta)) - vf) * (ft == 3)
            while abs(f) > 1e-6:
                eta = eta - f / np.mean(deta(x_phys.flatten(), eta, beta))
                f = np.mean(prj(x_phys, eta, beta)) - vf
            dHs = Hs / np.reshape(dprj(x_tilde, eta, beta), (ny, nx))
            x_phys = prj(x_phys, eta, beta)
        change = np.linalg.norm(x_phys - x_old) / np.sqrt(elem_num)
        x_old = x_phys.copy()
        #   ________________________________________________________________
        E, dE = ordered_simp_interpolation(x_phys, penalty, densities, elasticities, True)
        Ks = ((Ke[np.newaxis]).T * E).flatten(order='F')
        K = csc_matrix((Ks, (indexes[:, 0], indexes[:, 1])), shape=(dof, dof))[free, :][:, free].tocoo()
        u, b = np.zeros(dof), force[free]
        K = spmatrix(K.data, K.row.astype(np.int32), K.col.astype(np.int32))
        cholmod.linsolve(K, b)
        u[free] = np.array(b)[:, 0]
        #   ________________________________________________________________
        dC = np.reshape(-dE * np.sum((u[c_mat] @ Ke0) * u[c_mat], axis=1), (ny, nx), order='F')
        dC = correlate(dC / dHs, h, mode='reflect')[mask]
        P, dP = ordered_simp_interpolation(x_phys, 1 / penalty, densities, costs)
        P = (correlate(P, h, mode='reflect') / Hs)[mask]
        dP = correlate(dP / dHs, h, mode='reflect')[mask]
        #   ________________________________________________________________
        x[mask] = optimality_criterion(x[mask], dC, P, dP, vf, cf, max(0.15 * 0.96 ** loop, move))
        penalty, beta = cnt(penalty, penalCnt, loop), cnt(beta, betaCnt, loop)
        #   ________________________________________________________________
        print(f'Iteration = {loop}, Change = {change:0.5f}')
        plot(x_phys, loop, change, fig, ax, im, bg, densities)

    print(f'Model converged in {(time.time() - start):0.2f} seconds')
    ax.set_title(F'Iteration: {loop} | Model converged in {(time.time() - start):0.1f} seconds')
    plt.show(block=True)


def simp_interpolation(x, penal, Y_min, Y_max):
    y = Y_min + x ** penal * (Y_max - Y_min)
    dy = penal * (Y_max - Y_min) * x ** (penal - 1)
    return y.flatten(order='F'), dy.flatten(order='F')


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
    while abs((LV[1] - LV[0]) / (LV[1] + LV[0])) > 1e-6 or abs((LP[1] - LP[0]) / (LP[1] + LP[0])) > 1e-6:
        l_mid_v, l_mid_p = 0.5 * (LV[0] + LV[1]), 0.5 * (LP[0] + LP[1])
        B = -dC / (l_mid_v + l_mid_p * P + l_mid_p * x * dP)
        x_new = np.maximum(np.maximum(np.minimum(np.minimum(x * np.sqrt(np.abs(B)), xU), 1), xL), 1E-5)
        LV[0], LV[1] = (l_mid_v, LV[1]) if np.sum(x_new) > vf * x.size else (LV[0], l_mid_v)
        LP[0], LP[1] = (l_mid_p, LP[1]) if np.sum(x_new * P) / x.size > cf else (LP[0], l_mid_p)
    return x_new


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
    im = ax.imshow(x, origin='lower', cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
    ax.set_title(F'Iteration: {0}, Change: {1:0.4f}')
    ax.legend(custom_lines, names, ncol=len(colors))
    plt.get_current_fig_manager().window.showMaximized()
    plt.pause(0.1)
    bg = fig.canvas.copy_from_bbox(fig.bbox)
    ax.draw_artist(im)
    fig.canvas.blit(fig.bbox)
    return fig, ax, im, bg


def plot(x, loop, ch, fig, ax, im, bg, densities):
    array = x.copy()
    for i, d in enumerate(densities):
        array[(x > ((densities[i - 1] + d) / 2 if i > 0 else d))] = d
    fig.canvas.restore_region(bg)
    im.set_array(array)
    ax.set_title(F'Iteration: {loop}, Change: {ch:0.5f}')
    ax.draw_artist(im)
    fig.canvas.blit(fig.bbox)
    fig.canvas.flush_events()


def read_materials(input_path):
    material_df = pd.read_excel(input_path, sheet_name='Materials')
    D = material_df['Density'].tolist()
    E = material_df['Elasticity'].tolist()
    P = material_df['Cost'].tolist()
    M_name = material_df['Material'].tolist()
    M_color = material_df['Color'].tolist()
    return D, E, P, M_name, M_color


def read_options(input_path):
    options = pd.read_excel(input_path, sheet_name='Options')
    nx, ny, vf, cf, penalty, ft, max_it, r_min, eta, beta, move = options['Value']
    nx, ny, penalty, ft, max_it = np.array((nx, ny, penalty, ft, max_it), dtype=np.int32)
    return nx, ny, vf, cf, penalty, ft, max_it, r_min, eta, beta, move


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
