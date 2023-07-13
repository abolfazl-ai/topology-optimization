import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from utils import mesh, apply_bc, get_bc_load


def main(length, width, n_elx, n_ely, vol_frac, penalty):
    dof = 2 * (n_elx + 1) * (n_ely + 1)
    e_dof = np.zeros((n_elx * n_ely, 8), dtype=int)

    bc = get_bc_load(m=n_elx, n=n_ely)
    nodes, elements = mesh(length, width, n_elx, n_ely, bc)
    U, F = np.zeros(dof), np.zeros(dof)
    for n in nodes:
        U[2 * (n - 1):2 * (n - 1) + 2] = nodes[n].displacement
        F[2 * (n - 1):2 * (n - 1) + 2] = nodes[n].force

    for ii in range(n_elx):
        for jj in range(n_ely):
            el = jj + ii * n_ely
            nodes = elements[ii, jj].nodes
            e_dof[el, :] = 2 * (np.repeat(nodes, 2) - 1) + np.repeat([[0, 1]], 4, axis=0).flatten()
    iK = np.kron(e_dof, np.ones((8, 1))).flatten()
    jK = np.kron(e_dof, np.ones((1, 8))).flatten()

    KE = element_stiffness()
    x = vol_frac * np.ones(elements.shape)

    plt.ion()
    loop, change = 0, 1
    while change > 0.001:
        loop += 1
        x_old = x.copy()
        Un = fem(iK, jK, KE, U, F, x, penalty, dof)
        ce = (np.dot(Un[e_dof].reshape(n_elx * n_ely, 8), KE) * Un[e_dof].reshape(n_elx * n_ely, 8)).sum(1)
        dc = ((-penalty * x.flatten() ** (penalty - 1)) * ce).reshape(x.shape)

        x = oc(n_elx, n_ely, vol_frac, x, dc)
        change = np.max(abs(x - x_old))

        print(F'Iteration: {loop}, Change: {change}')
        plt.imshow(-x.T, cmap='gray', vmin=-1, vmax=0, origin='lower')
        plt.pause(1E-6)

    print('Model converged')
    plt.pause(1E6)


def fem(iK, jK, KE, U, F, x, penalty, dof):
    elem_k = ((KE.flatten()[np.newaxis]).T * (x.flatten() ** penalty)).flatten(order='F')
    K = coo_matrix((elem_k, (iK, jK)), shape=(dof, dof)).tocsc()
    U = apply_bc(K, U, F)
    return U


def oc(n_elx, n_ely, vol_frac, x, dc):
    l1, l2, move = 0, 100000, 0.2
    x_new = np.zeros(x.shape)
    while (l2 - l1) / (l1 + l2) > 1e-3:
        l_mid = 0.5 * (l2 + l1)
        x_new = np.maximum(0.001, np.maximum(x - move,
                                             np.minimum(1.0,
                                                        np.minimum(x + move,
                                                                   x * np.sqrt(-dc / l_mid)))))
        if np.sum(x_new) - vol_frac * n_elx * n_ely > 0:
            l1 = l_mid
        else:
            l2 = l_mid
    return x_new


def element_stiffness(E=1, nu=0.3):
    k = np.array([1 / 2 - nu / 6, 1 / 8 + nu / 8, -1 / 4 - nu / 12,
                  -1 / 8 + 3 * nu / 8, -1 / 4 + nu / 12, -1 / 8 - nu / 8,
                  nu / 6, 1 / 8 - 3 * nu / 8])
    KE = E / (1 - nu ** 2) * np.array([[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                                       [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                                       [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                                       [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                                       [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                                       [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                                       [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                                       [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]])
    return KE


main(1, 1, 100, 100, 0.3, 3)
