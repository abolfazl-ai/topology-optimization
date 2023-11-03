import numpy as np
from scipy.ndimage import correlate
from finite_element import FEM


class TopOpt:

    def __init__(self, mesh, bc, fil, opt, materials, pres, mask, iter_callback):
        self.materials, self.pres, self.mask, self.iter_callback = materials, pres, mask, iter_callback
        self.mesh, self.bc, self.fil, self.opt = mesh, bc, fil, opt
        self.vf, self.shape = opt['volume_fraction'], mesh['shape']
        self.h, self.hs, self.dhs = self._prepare_filter(fil['radius'])
        self.x, self.de, self.dw = pres.copy(), np.zeros(self.shape), np.zeros(self.shape)
        self.c, self.w, self.c_change, self.x_change, self.loop = [], [], 1, 1, 0
        self.fem = FEM(mesh, bc)
        self.dw[mask] = 1 / (self.mesh['elem_num'] * self.vf)
        self.x[mask] = (self.vf * (self.mesh['elem_num'] - pres[~mask].size)) / pres[mask].size

    def _prepare_filter(self, r):
        ds = np.meshgrid(*(np.arange(-np.ceil(r) + 1, np.ceil(r)) for _ in self.mesh['shape']))
        h = np.maximum(0, r - np.sqrt(np.sum((n ** 2 for n in ds), axis=0)))
        hs = correlate(np.ones(self.shape), h, mode=self.fil['filter_bc'])
        return h, hs, hs.copy()

    def _apply_filter(self, x):
        ft, filter_bc, r, eta, beta = [self.fil[key] for key in ('filter', 'filter_bc', 'radius', 'eta', 'beta')]
        x_phys, x_tilde = x.copy(), correlate(x, self.h, mode=filter_bc) / self.hs
        x_phys[self.mask] = x_tilde[self.mask]
        if ft > 1:
            f = (np.mean(prj(x_phys, eta, beta)) - self.vf) * (ft == 3)
            while abs(f) > 1e-6:
                eta = eta - f / np.mean(deta(x_phys.flatten(), eta, beta))
                f = np.mean(prj(x_phys, eta, beta)) - self.vf
            self.dhs = self.hs / np.reshape(dprj(x_tilde, eta, beta), self.shape)
            x_phys[self.mask] = prj(x_phys, eta, beta)[self.mask]
        return x_phys

    def _compliance(self, x, c_mat):
        e, de = ordered_simp(x, self.opt['penalty'], self.materials['D'], self.materials['E'], True)
        u = self.fem.solve(e)
        c = np.sum(e * np.sum((u[c_mat] @ self.fem.k) * u[c_mat], 1)) / np.prod(self.shape)
        dc = np.reshape(-de * np.sum((u[c_mat] @ self.fem.k) * u[c_mat], 1), self.shape, 'F')
        dc = correlate(dc / self.dhs, self.h, mode=self.fil['filter_bc'])
        w, dw = np.mean(x), correlate(self.dw / self.dhs, self.h, mode=self.fil['filter_bc'])
        return c, dc, w, dw

    def _optimality_criterion(self, x, dc, dw):
        x_new, xT = x.copy(), x[self.mask]
        xU, xL = xT + self.opt['move'], xT - self.opt['move']
        ocP = xT * np.real(np.sqrt(-dc[self.mask] / dw[self.mask]))
        LM = [0, np.mean(ocP) / self.vf]
        while abs((LM[1] - LM[0]) / (LM[1] + LM[0])) > 1e-4:
            l_mid = 0.5 * (LM[0] + LM[1])
            x_new[self.mask] = np.maximum(np.minimum(np.minimum(ocP / l_mid, xU), 1), xL)
            LM[0], LM[1] = (l_mid, LM[1]) if np.mean(x_new) > self.vf else (LM[0], l_mid)
        return x_new

    def optimize(self):
        while (self.c_change > self.opt['c_conv'] and
               self.x_change > self.opt['x_conv'] and
               self.loop < self.opt['max_it']):
            self.loop += 1
            x_old = self.x.copy()
            x = self._apply_filter(self.x)
            c, dc, w, dw = self._compliance(x, self.mesh['c_mat'])
            self.w.append(w), self.c.append(c)
            self.x[self.mask] = self._optimality_criterion(self.x, dc, dw)[self.mask]
            self.c_change, self.x_change = (self.opt['move'], self.opt['move']) if self.loop < 3 else (
                abs(np.sqrt((self.c[-2] - self.c[-1]) ** 2 + (self.c[-3] - self.c[-2]) ** 2) / np.max(np.abs(self.c))),
                np.linalg.norm(x_old - self.x) / np.sqrt(self.mesh['elem_num']))
            self.iter_callback(self.loop, x, self.x_change, self.c_change, c, w)
            self.opt['penalty'] = self.opt['penalty'] + self.opt['penalty_increase']


def ordered_simp(x, penalty, X, Y, flatten=False):
    y, dy = np.ones(x.shape), np.zeros(x.shape)
    for i in range(len(X) - 1):
        mask = ((X[i] < x) if i > 0 else True) & (x < X[i + 1] if i < len(X) - 2 else True)
        A = (Y[i] - Y[i + 1]) / (X[i] ** penalty - X[i + 1] ** penalty)
        B = Y[i] - A * (X[i] ** penalty)
        y[mask] = A * (x[mask] ** penalty) + B
        dy[mask] = A * penalty * (x[mask] ** (penalty - 1))
    return (y.flatten(order='F'), dy.flatten(order='F')) if flatten else (y, dy)


def prj(v, eta, beta):
    return (np.tanh(beta * eta) + np.tanh(beta * (v - eta))) / (np.tanh(beta * eta) + np.tanh(beta * (1 - eta)))


def deta(v, e, b):
    return -b * (1 / np.sinh(b)) * ((1 / np.cosh(b * (v - e))) ** 2) * (np.sinh(v * b)) * (np.sinh((1 - v) * b))


def dprj(v, eta, beta):
    return beta * (1 - np.tanh(beta * (v - eta)) ** 2) / (np.tanh(beta * eta) + np.tanh(beta * (1 - eta)))
