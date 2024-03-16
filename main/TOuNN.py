import numpy as np
from scipy.ndimage import correlate
from finite_element import FEM
import torch
import torch.nn as nn


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

    def setDevice(self):
        return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    def _compliance(self, x, c_mat):
        e, de = ordered_simp(x, self.opt['penalty'], self.materials['D'], self.materials['E'], True)
        u = self.fem.solve(e)
        c = np.sum(e * np.sum((u[c_mat] @ self.fem.k) * u[c_mat], 1)) / np.prod(self.shape)
        dc = np.reshape(-de * np.sum((u[c_mat] @ self.fem.k) * u[c_mat], 1), self.shape, 'F')
        dc = correlate(dc / self.dhs, self.h, mode=self.fil['filter_bc'])
        w, dw = np.mean(x), correlate(self.dw / self.dhs, self.h, mode=self.fil['filter_bc'])
        return c, dc, w, dw

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


class TopOptNN(nn.Module):
    def __init__(self, input_dim, output_dim, layers):
        self.input_dim, self.output_dim, self.layers = input_dim, output_dim, layers
        super().__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(nn.Linear(input_dim, layers[0]), nn.LeakyReLU())
        for i in range(0, len(layers)):
            self.network.append(nn.Linear(layers[i], layers[i + 1]))
            self.network.append(nn.LeakyReLU())
        self.network.append(nn.Linear(layers[-1], output_dim))
        self.network.append(nn.Softmax(dim=1))

    def forward(self, x):
        return self.network(x)
