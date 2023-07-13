import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from scipy.sparse import coo_matrix
from utils import mesh, apply_bc, get_bc_load, get_materials


def main(length, width, n_elx, n_ely, X0, r_min, vol_frac, cost_frac, penalty, MinMove):
    dof = 2 * (n_elx + 1) * (n_ely + 1)
    e_dof = np.zeros((n_elx * n_ely, 8), dtype=int)

    D, E, P, nu, M_name, M_color = get_materials()
    levels = [D[0]] + [0.5 * (D[i] + D[i + 1]) for i in range(len(D) - 1)] + [D[-1]]
    bc = get_bc_load(m=n_elx, n=n_ely, input_path='material-bc-load2.xlsx')

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
    while change > 0.01:
        loop += 1
        x_old = x.copy()

        E_, dE_ = ordered_simp_interpolation(n_elx, n_ely, x, penalty, D, E)
        P_, dP_ = ordered_simp_interpolation(n_elx, n_ely, x, 1 / penalty, D, P)

        Un = fem(iK, jK, KE, U, F, E_, penalty, dof)
        ce = (np.dot(Un[e_dof].reshape(n_elx * n_ely, 8), KE) * Un[e_dof].reshape(n_elx * n_ely, 8)).sum(1)
        dc = (-dE_.flatten() * ce).reshape(x.shape)

        dc = check(n_elx, n_ely, r_min, x, dc)
        x = oc(n_elx, n_ely, vol_frac, x, cost_frac, dc, P_, dP_, loop, MinMove)
        change = np.max(abs(x - x_old))

        print(F'Iteration: {loop}, Change: {change}')
        cs = plt.contourf(x.T, levels=levels, colors=M_color)
        proxy = [plt.Rectangle((0, 0), 1, 1, fc=pc.get_facecolor()[0]) for pc in cs.collections]
        plt.legend(proxy, M_name)
        plt.pause(1E-6)

    print('Model converged')
    plt.pause(1E6)


def fem(iK, jK, KE, U, F, E_int, penalty, dof):
    elem_k = ((KE.flatten()[np.newaxis]).T * E_int.flatten()).flatten(order='F')
    K = coo_matrix((elem_k, (iK, jK)), shape=(dof, dof)).tocsc()
    U = apply_bc(K, U, F)
    return U


def ordered_simp_interpolation(n_elx, n_ely, x, penal, X, Y):
    y = np.zeros(x.shape)
    dy = np.zeros(x.shape)

    for i in range(n_ely):
        for j in range(n_elx):
            for k in range(len(X) - 1):
                if (X[k] < x[j, i]) and (X[k + 1] >= x[j, i]):
                    A = (Y[k] - Y[k + 1]) / (X[k] ** (1 * penal) - X[k + 1] ** (1 * penal))
                    B = Y[k] - A * (X[k] ** (1 * penal))
                    y[j, i] = A * (x[j, i] ** (1 * penal)) + B
                    dy[j, i] = A * penal * (x[j, i] ** ((1 * penal) - 1))
                    break
    return y, dy


def check(n_elx, n_ely, r_min, x, dc):
    dcn = np.zeros(dc.shape)
    for i in range(n_elx):
        for j in range(n_ely):
            s = 0.0
            for k in range(max(i - int(np.floor(r_min)), 0), min(i + int(np.ceil(r_min)), n_elx)):
                for l in range(max(j - int(np.floor(r_min)), 0), min(j + int(np.ceil(r_min)), n_ely)):
                    fac = r_min - np.sqrt((i - k) ** 2 + (j - l) ** 2)
                    s += max(0, fac)
                    dcn[i, j] += max(0, fac) * x[k, l] * dc[k, l]
            dcn[i, j] /= x[i, j] * s
    return dcn


def oc(n_elx, n_ely, vol_frac, x, cost_frac, dc, P_, dP_, loop, MinMove):
    Temp = -dc / (P_ + x * dP_)
    lV1, lV2, lP1, lP2 = 0.0, 2 * np.max(-dc), 0.0, 2 * np.max(Temp)
    move = max(0.15 * 0.96 ** loop, MinMove)
    x_new = np.zeros(x.shape)
    while ((lV2 - lV1) / (lV1 + lV2) > 1e-6) or ((lP2 - lP1) / (lP1 + lP2) > 1e-6):
        lmidV = 0.5 * (lV2 + lV1)
        lmidP = 0.5 * (lP2 + lP1)
        Temp = lmidV + lmidP * P_ + lmidP * x * dP_
        Coef = -dc / Temp
        Coef = np.abs(Coef)
        x_new = np.maximum(10 ** -5, np.maximum(x - move, np.minimum(1., np.minimum(x + move, x * np.sqrt(Coef)))))
        if np.sum(x_new) - vol_frac * n_elx * n_ely > 0:
            lV1 = lmidV
        else:
            lV2 = lmidV

        CurrentCostFrac = np.sum(x_new * P_) / (n_elx * n_ely)
        if CurrentCostFrac - cost_frac > 0:
            lP1 = lmidP
        else:
            lP2 = lmidP

    return x_new


def element_stiffness(E=1, nu=0.3):
    A11 = np.array([[12, 3, -6, -3], [3, 12, 3, 0], [-6, 3, 12, -3], [-3, 0, -3, 12]])
    A12 = np.array([[-6, -3, 0, 3], [-3, -6, -3, -6], [0, -3, -6, 3], [3, -6, 3, -6]])
    B11 = np.array([[-4, 3, -2, 9], [3, -4, -9, 4], [-2, -9, -4, -3], [9, 4, -3, -4]])
    B12 = np.array([[2, -3, 4, -9], [-3, 2, 9, -2], [4, 9, 2, 3], [-9, -2, 3, 2]])
    KE = E / (1 - nu ** 2) / 24 * (np.block([[A11, A12], [A12.T, A11]]) + nu * np.block([[B11, B12], [B12.T, B11]]))
    return KE


main(1, 1, 100, 50, 0.5, 2.5, 0.3, 0.4, 3, 0.001)
