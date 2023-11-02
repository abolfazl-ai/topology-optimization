import numpy as np
from scipy.ndimage import correlate
from finite_element import FEM


def prj(v, eta, beta):
    return (np.tanh(beta * eta) + np.tanh(beta * (v - eta))) / (np.tanh(beta * eta) + np.tanh(beta * (1 - eta)))


def deta(v, e, b):
    return -b * (1 / np.sinh(b)) * ((1 / np.cosh(b * (v - e))) ** 2) * (np.sinh(v * b)) * (np.sinh((1 - v) * b))


def dprj(v, eta, beta):
    return beta * (1 - np.tanh(beta * (v - eta)) ** 2) / (np.tanh(beta * eta) + np.tanh(beta * (1 - eta)))


def ordered_simp(x, penalty, X, Y, flatten=False):
    y, dy = np.ones(x.shape), np.zeros(x.shape)
    for i in range(len(X) - 1):
        mask = ((X[i] < x) if i > 0 else True) & (x < X[i + 1] if i < len(X) - 2 else True)
        A = (Y[i] - Y[i + 1]) / (X[i] ** penalty - X[i + 1] ** penalty)
        B = Y[i] - A * (X[i] ** penalty)
        y[mask] = A * (x[mask] ** penalty) + B
        dy[mask] = A * penalty * (x[mask] ** (penalty - 1))
    return y.flatten(order='F') if flatten else y, dy.flatten(order='F') if flatten else dy


class TopOpt:

    def __init__(self, mesh, bc, fil, opt, materials, pres, mask, iter_callback):
        self.mesh, self.bc, self.fil, self.opt = mesh, bc, fil, opt
        (self.vf, self.p, self.move, self.c_conv, self.x_conv, self.max_it) = \
            [opt[key] for key in ('volume_fraction', 'penalty', 'move', 'c_conv', 'x_conv', 'max_it')]
        self.ft, self.filter_bc = [fil[key] for key in ('filter', 'filter_bc')]
        self.r, self.eta, self.beta = [fil[key] for key in ('radius', 'eta', 'beta')]
        self.shape, self.c_mat, self.elem_num = [mesh[key] for key in ('shape', 'c_mat', 'elem_num')]
        self.materials, self.pres, self.mask, self.iter_callback = materials, pres, mask, iter_callback
        self.fem = FEM(mesh, bc)
        self.h, self.hs, self.dhs = self.prepare_filter(self.r)
        self.x, self.de, self.dw = pres.copy(), np.zeros(self.shape), np.zeros(self.shape)
        self.c, self.w, self.c_change, self.x_change, self.loop = [], [], 1, 1, 0
        self.dw[mask] = 1 / (self.elem_num * self.vf)
        self.x[mask] = (self.vf * (self.elem_num - pres[~mask].size)) / pres[mask].size

    def prepare_filter(self, r):
        dy, dz, dx = np.meshgrid(np.arange(-np.ceil(r) + 1, np.ceil(r)),
                                 np.arange(-np.ceil(r) + 1, np.ceil(r)),
                                 np.arange(-np.ceil(r) + 1, np.ceil(r)))
        h = np.maximum(0, r - np.sqrt(dx ** 2 + dy ** 2 + dz ** 2))
        hs = correlate(np.ones(self.shape), h, mode=self.filter_bc)
        return h, hs, hs.copy()

    def apply_filter(self, x):
        x_phys, x_tilde = x.copy(), correlate(x, self.h, mode=self.filter_bc) / self.hs
        x_phys[self.mask] = x_tilde[self.mask]
        if self.ft > 1:
            f = (np.mean(prj(x_phys, self.eta, self.beta)) - self.vf) * (self.ft == 3)
            while abs(f) > 1e-6:
                self.eta = self.eta - f / np.mean(deta(x_phys.flatten(), self.eta, self.beta))
                f = np.mean(prj(x_phys, self.eta, self.beta)) - self.vf
            self.dhs = self.hs / np.reshape(dprj(x_tilde, self.eta, self.beta), self.shape)
            x_phys[self.mask] = prj(x_phys, self.eta, self.beta)[self.mask]
        return x_phys

    def compliance(self, x):
        e, de = ordered_simp(x, self.p, self.materials['D'], self.materials['E'], True)
        self.fem.update_stiffness(e)
        u = self.fem.solve()
        c = np.sum(e * np.sum((u[self.c_mat] @ self.fem.k) * u[self.c_mat], axis=1)) / np.prod(self.shape)
        dc = np.reshape(-de * np.sum((u[self.c_mat] @ self.fem.k) * u[self.c_mat], axis=1), self.shape, order='F')
        dc = correlate(dc / self.dhs, self.h, mode=self.filter_bc)
        w, dw = np.mean(x), correlate(self.dw / self.dhs, self.h, mode=self.filter_bc)
        return c, dc, w, dw

    def optimality_criterion(self, x, dc, dw):
        x_new, xT = x.copy(), x[self.mask]
        xU, xL = xT + self.move, xT - self.move
        ocP = xT * np.real(np.sqrt(-dc[self.mask] / dw[self.mask]))
        LM = [0, np.mean(ocP) / self.vf]
        while abs((LM[1] - LM[0]) / (LM[1] + LM[0])) > 1e-4:
            l_mid = 0.5 * (LM[0] + LM[1])
            x_new[self.mask] = np.maximum(np.minimum(np.minimum(ocP / l_mid, xU), 1), xL)
            LM[0], LM[1] = (l_mid, LM[1]) if np.mean(x_new) > self.vf else (LM[0], l_mid)
        return x_new

    def optimize(self):
        while self.c_change > self.c_conv and self.x_change > self.x_conv and self.loop < self.max_it:
            self.loop += 1
            x_old = self.x.copy()
            x = self.apply_filter(self.x)
            c, dc, w, dw = self.compliance(x)
            self.x[self.mask] = self.optimality_criterion(self.x, dc, dw)[self.mask]
            self.w.append(w), self.c.append(c)
            self.c_change = self.move if self.loop < 3 else (
                abs(np.sqrt((self.c[-2] - self.c[-1]) ** 2 + (self.c[-3] - self.c[-2]) ** 2) / np.max(np.abs(self.c))))
            self.x_change = self.move if self.loop < 3 else np.linalg.norm(x_old - self.x) / np.sqrt(self.elem_num)
            self.iter_callback(self.loop, x, self.x_change, self.c_change, c, w)

        return self.x, self.c, self.w
