import numpy as np
from scipy.ndimage import correlate
from scipy.sparse import csr_array, diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import time


def main(nx, ny, vol_f, penal, r_min, ft, eta, beta, move, max_it):
    E_min, E_max, nu = 1E-9, 1.0, 0.3
    penalCnt = [1, 1, 25, 0.25]
    betaCnt = [1, 1, 25, 2]
    #   ________________________________________________________________
    nEl, nDof = nx * ny, (1 + ny) * (1 + nx) * 2
    Ke, Ke0, cMat, Iar = element_stiffness(nx, ny, nu)
    #   ________________________________________________________________
    fixed = np.union1d(np.arange(0, 2 * (ny + 1), 2), 2 * (1 + nx) * (1 + ny) - 1)
    pasS, pasV = [], []
    F = csr_array(([-1], ([1], [0])), shape=(nDof, 1))
    free = np.setdiff1d(np.arange(0, nDof), fixed)
    act = np.setdiff1d(np.arange(0, nEl), np.union1d(pasS, pasV))
    #   ________________________________________________________________
    dy, dx = np.meshgrid(np.arange(-np.ceil(r_min) + 1, np.ceil(r_min)),
                         np.arange(-np.ceil(r_min) + 1, np.ceil(r_min)))
    h = np.maximum(0, r_min - np.sqrt(dx ** 2 + dy ** 2))
    Hs = correlate(np.ones((ny, nx)), h, mode='reflect')
    dHs = Hs
    #   ________________________________________________________________
    x, dsK, dV = np.zeros(nEl), np.zeros(nEl), np.zeros(nEl)
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
        xPhys[act] = np.reshape(xTilde, (-1,), 'F')[act]
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
        sK = E_min + xPhys ** penal * (E_max - E_min)
        dsK[act] = -penal * (E_max - E_min) * xPhys[act] ** (penal - 1)
        sK = np.reshape(Ke.reshape((-1, 1)) * sK, (len(Ke) * nEl,), order='F')
        K = csr_array((sK, (Iar[:, 0], Iar[:, 1])), shape=(nDof, nDof))
        U, K_free = np.zeros(nDof), K[free, :][:, free]
        K_free += K_free.T - diags(K_free.diagonal())
        U[free] = spsolve(K_free, F[free])
        #   ________________________________________________________________
        dc = dsK * np.sum((U[cMat] @ Ke0) * U[cMat], axis=1)
        dc = np.reshape(correlate(np.reshape(dc, (ny, nx), order='F') /
                                  dHs, h, mode='reflect'), (-1,), 'F')
        dV0 = np.reshape(correlate(np.reshape(dV, (ny, nx), 'F') /
                                   dHs, h, mode='reflect'), (-1,), 'F')
        #   ________________________________________________________________
        xT = x[act]
        xU, xL = xT + move, xT - move
        ocP = xT * np.real(np.sqrt(-dc[act] / dV0[act]))
        LM = [0, np.mean(ocP) / vol_f]
        while abs((LM[1] - LM[0]) / (LM[1] + LM[0])) > 1e-4:
            l_mid = 0.5 * (LM[0] + LM[1])
            x[act] = np.maximum(np.minimum(np.minimum(ocP / l_mid, xU), 1), xL)
            if np.mean(x) > vol_f:
                LM[0] = l_mid
            else:
                LM[1] = l_mid
        penal, beta = cnt(penal, penalCnt, loop), cnt(beta, betaCnt, loop)
        #   ________________________________________________________________
        print(f'It = {loop}, Change = {ch}')
        plot(1 - np.reshape(xPhys, (ny, nx), 'F'), loop, ch, fig, ax, im, bg)
    print(f'Model converged in {(time.time() - start):0.2f} Seconds')
    plt.show(block=True)


def prj(v, eta, beta):
    return (np.tanh(beta * eta) + np.tanh(beta * (v - eta))) / (np.tanh(beta * eta) + np.tanh(beta * (1 - eta)))


def deta(v, eta, beta):
    return -beta * (1 / np.sinh(beta)) * ((1 / np.cosh(beta * (v - eta))) ** 2) * (np.sinh(v * beta)) * (
        np.sinh((1 - v) * beta))


def dprj(v, eta, beta):
    return beta * (1 - np.tanh(beta * (v - eta)) ** 2) / (np.tanh(beta * eta) + np.tanh(beta * (1 - eta)))


def cnt(v, vCnt, el):
    condition = (el >= vCnt[0]) and (v < vCnt[1]) and (el % vCnt[2] == 0)
    return v + condition * vCnt[3]


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
    ax.set_title(F'Iteration: {loop}, Change: {ch:0.4f}')
    ax.draw_artist(im)
    fig.canvas.blit(fig.bbox)
    fig.canvas.flush_events()


main(300, 100, 0.2, 3, 8.75, 1, 0.5, 2, 0.2, 500)
