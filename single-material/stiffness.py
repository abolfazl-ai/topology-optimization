import numpy as np
from scipy.sparse import lil_matrix


def numerical_integral(f, a, b, c, d, n):
    # Gauss-Legendre Quadrature weights and nodes
    nodes, weights = np.polynomial.legendre.leggauss(n)
    integral = 0.0
    for i in range(n):
        for j in range(n):
            x = 0.5 * (b - a) * nodes[i] + 0.5 * (b + a)
            y = 0.5 * (d - c) * nodes[j] + 0.5 * (d + c)
            integral += weights[i] * weights[j] * f(x, y)

    integral *= 0.25 * (b - a) * (d - c)
    return integral


def stress_strain_matrix(stress_mode, E, v):
    if stress_mode == 0:
        D = (E / (1 - v ** 2)) * np.matrix([[1, v, 0], [v, 1, 0], [0, 0, (1 - v) / 2]])
    else:
        D = (E / ((1 + v) * (1 - 2 * v))) * np.matrix([[1 - v, v, 0], [v, 1 - v, 0], [0, 0, (1 - 2 * v) / 2]])
    return D


def q4_j_determinant(X, Y, s, t):
    return (X.T @ np.array([[0, 1 - t, t - s, s - 1],
                            [t - 1, 0, s + 1, -s - t],
                            [s - t, -s - 1, 0, t + 1],
                            [1 - s, s + t, -t - 1, 0]]) @ Y) / 8


def q4_b_matrix(X, Y, s, t):
    N_s = np.array([t - 1, 1 - t, 1 + t, -1 - t]) / 4
    N_t = np.array([s - 1, -s - 1, 1 + s, 1 - s]) / 4
    j_det = q4_j_determinant(X, Y, s, t)
    NX = np.array([(N_s @ X), -(N_t @ X)]) @ np.array([N_t, N_s])
    NY = np.array([(N_t @ Y), -(N_s @ Y)]) @ np.array([N_s, N_t])
    return np.array([[NY[0], 0, NY[1], 0, NY[2], 0, NY[3], 0],
                     [0, NX[0], 0, NX[1], 0, NX[2], 0, NX[3]],
                     [NX[0], NY[0], NX[1], NY[1], NX[2], NY[2], NX[3], NY[3]]]) / j_det


def q4_k_st(D, X, Y, s, t, h):
    B = q4_b_matrix(X, Y, s, t)
    j_det = q4_j_determinant(X, Y, s, t)
    return h * j_det * ((B.T @ D) @ B)


def q4_k_matrix(element, all_nodes):
    D = stress_strain_matrix(element.stress_mode, element.young, element.poisson)
    X, Y = element.get_xy(all_nodes)
    K = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            K[i, j] = numerical_integral(lambda s, t: q4_k_st(D, X, Y, s, t, element.thickness)[i, j], -1, 1, -1, 1, 2)
    return K


def assemble(elements, k_local, dof):
    K = lil_matrix((sum(dof.values()), sum(dof.values())))
    for ii in range(elements.shape[1]):
        for jj in range(elements.shape[0]):
            e_k = k_local[ii, jj]  # Stiffness matrix of the element
            nodes = elements[ii, jj].nodes  # Numbers of nodes of the element in clockwise order
            diff = {m: sum(dof[nodes[i]] for i in range(m)) for m in range(len(nodes))}
            for i, node_i in enumerate(nodes):
                ni_dof = dof[node_i]  # Node i degrees of freedom
                row = sum(d for n, d in dof.items() if n < node_i)  # The node's start row in global matrix
                for j, node_j in enumerate(nodes):
                    nj_dof = dof[node_j]  # Node j degrees of freedom
                    col = sum(d for n, d in dof.items() if n < node_j)  # The node's start column in global matrix
                    K[row:row + ni_dof, col:col + nj_dof] += e_k[diff[i]:diff[i] + ni_dof, diff[j]:diff[j] + nj_dof]
    return K
