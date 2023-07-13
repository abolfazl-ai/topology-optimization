import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from utils import mesh, apply_bc, create_load_bc


def main(length, width, n_elx, n_ely, X0, r_min, vol_frac, cost_frac, penalty, D, E, P, MColor, MName, MinMove):
    dof = 2 * (n_elx + 1) * (n_ely + 1)
    e_dof = np.zeros((n_elx * n_ely, 8), dtype=int)

    bc = create_load_bc(m=n_elx, n=n_ely)
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
    x = X0 * np.ones(elements.shape)

    plt.ion()
    loop, change = 0, 1
    while change > 0.001:
        loop += 1
        x_old = x.copy()

        E_, dE_ = ordered_simp_interpolation(n_elx, n_ely, x, penalty, D, E)
        P_, dP_ = ordered_simp_interpolation(n_elx, n_ely, x, 1 / penalty, D, P)

        Un = fem(iK, jK, KE, U, F, E_, penalty, dof)
        ce = (np.dot(Un[e_dof].reshape(n_elx * n_ely, 8), KE) * Un[e_dof].reshape(n_elx * n_ely, 8)).sum(1)
        dc = (-dE_.flatten() * ce).reshape(x.shape)

        x = oc(n_elx, n_ely, vol_frac, x, dc)
        change = np.max(abs(x - x_old))

        print(F'Iteration: {loop}, Change: {change}')
        plt.imshow(-x.T, cmap='gray', vmin=-1, vmax=0, origin='lower')
        plt.pause(1E-6)

    print('Model converged')
    plt.pause(1E6)


def fem(iK, jK, KE, U, F, E_int, penalty, dof):
    elem_k = ((KE.flatten()[np.newaxis]).T * (E_int.flatten() ** penalty)).flatten(order='F')
    K = coo_matrix((elem_k, (iK, jK)), shape=(dof, dof)).tocsc()
    U = apply_bc(K, U, F)
    return U


def ordered_simp_interpolation(n_elx, n_ely, x, penal, X, Y):
    y = np.zeros((n_ely, n_elx))
    dy = np.zeros((n_ely, n_elx))

    for i in range(n_elx):
        for j in range(n_ely):
            for k in range(len(X) - 1):
                if (X[k] < x[j, i]) and (X[k + 1] >= x[j, i]):
                    A = (Y[k] - Y[k + 1]) / (X[k] ** (1 * penal) - X[k + 1] ** (1 * penal))
                    B = Y[k] - A * (X[k] ** (1 * penal))
                    y[j, i] = A * (x[j, i] ** (1 * penal)) + B
                    dy[j, i] = A * penal * (x[j, i] ** ((1 * penal) - 1))
                    break
    return y, dy


def oc(n_elx, n_ely, vol_frac, x, dc):
    l1, l2, move = 0, 100000, 0.2
    x_new = np.zeros(x.shape)
    while (l2 - l1) / (l1 + l2) > 1e-3:
        l_mid = 0.5 * (l2 + l1)
        x_new[:] = np.maximum(0.001, np.maximum(x - move,
                                                np.minimum(1.0,
                                                           np.minimum(x + move,
                                                                      x * np.sqrt(-dc / l_mid)))))
        if np.concatenate(x_new).sum() - vol_frac * n_elx * n_ely > 0:
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


D = [0, 0.4, 0.7, 1.0]
E = [0, 0.2, 0.6, 1.0]
P = [0, 0.5, 0.8, 1.0]
MName = ['Void' 'A' 'B' 'C']
MColor = ['w', 'b', 'r', 'k']
main(1, 1, 200, 200, 0.5, 2.5, 0.3,
     0.4, 3, D, E, P, MColor, MName, 0.001)
