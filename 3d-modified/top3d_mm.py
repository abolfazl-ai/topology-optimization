import numpy as np
from scipy.ndimage import correlate
from finite_element import FEM


def top3d_mm(mesh, bc, fil, opt, materials, pres, mask, iter_callback):
    fem = FEM(mesh, bc)
    h, Hs, dHs = prepare_filter(mesh['shape'], fil)
    #   ________________________________________________________________
    x, de, dw, c, w = pres.copy(), np.zeros(mesh['shape']), np.zeros(mesh['shape']), [], []
    dw[mask] = 1 / (mesh['elem_num'] * opt['volume_fraction'])
    x[mask] = (opt['volume_fraction'] * (mesh['elem_num'] - pres[~mask].size)) / pres[mask].size
    x_phys, x_old, c_change, x_change, loop = x.copy(), x.copy(), 1, 1, 0
    while c_change > opt['c_conv'] and x_change > opt['x_conv'] and loop < opt['max_iteration']:
        loop += 1
        x_tilde = correlate(x, h, mode=fil['filter_bc']) / Hs
        x_phys[mask] = x_tilde[mask]
        if fil['filter'] > 1:
            f = (np.mean(prj(x_phys, fil['eta'], fil['beta'])) - opt['volume_fraction']) * (fil['filter'] == 3)
            while abs(f) > 1e-6:
                fil['eta'] = fil['eta'] - f / np.mean(deta(x_phys.flatten(), fil['eta'], fil['beta']))
                f = np.mean(prj(x_phys, fil['eta'], fil['beta'])) - opt['volume_fraction']
            dHs = Hs / np.reshape(dprj(x_tilde, fil['eta'], fil['beta']), mesh['shape'])
            x_phys[mask] = prj(x_phys, fil['eta'], fil['beta'])[mask]
        #   ________________________________________________________________
        e, de = ordered_simp_interpolation(x_phys, opt['penalty'], materials['D'], materials['E'], True)
        fem.update_stiffness(e)
        u = fem.solve()
        #   ________________________________________________________________
        dc = np.reshape(-de * np.sum((u[mesh['c_mat']] @ fem.k) * u[mesh['c_mat']], axis=1), mesh['shape'], order='F')
        dc = correlate(dc / dHs, h, mode=fil['filter_bc'])
        dws = correlate(dw / dHs, h, mode=fil['filter_bc'])
        x[mask] = optimality_criterion(x, opt['volume_fraction'], opt['move'], dc, dws, mask)[mask]
        #   ________________________________________________________________
        w.append(np.mean(x))
        c.append(np.sum(e * np.sum((u[mesh['c_mat']] @ fem.k) * u[mesh['c_mat']], axis=1)) / np.prod(mesh['shape']))
        c_change = opt['move'] if loop < 3 else abs(np.sqrt((c[-2] - c[-1]) ** 2 + (c[-3] - c[-2]) ** 2) / c[0])
        x_change = opt['move'] if loop < 3 else np.linalg.norm(x_phys - x_old) / np.sqrt(mesh['elem_num'])
        x_old = x_phys.copy()
        iter_callback(loop, x_phys, x_change, c_change, c[-1], w[-1])
    return x_phys, c, w


def prj(v, eta, beta):
    return (np.tanh(beta * eta) + np.tanh(beta * (v - eta))) / (np.tanh(beta * eta) + np.tanh(beta * (1 - eta)))


def deta(v, eta, beta):
    return (-beta * (1 / np.sinh(beta)) * ((1 / np.cosh(beta * (v - eta))) ** 2) *
            (np.sinh(v * beta)) * (np.sinh((1 - v) * beta)))


def dprj(v, eta, beta):
    return beta * (1 - np.tanh(beta * (v - eta)) ** 2) / (np.tanh(beta * eta) + np.tanh(beta * (1 - eta)))


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


def prepare_filter(shape, filter_params):
    r = filter_params['radius']
    dy, dz, dx = np.meshgrid(np.arange(-np.ceil(r) + 1, np.ceil(r)),
                             np.arange(-np.ceil(r) + 1, np.ceil(r)),
                             np.arange(-np.ceil(r) + 1, np.ceil(r)))
    h = np.maximum(0, r - np.sqrt(dx ** 2 + dy ** 2 + dz ** 2))
    Hs = correlate(np.ones(shape), h, mode=filter_params['filter_bc'])
    return h, Hs, Hs.copy()
