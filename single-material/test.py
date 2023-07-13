import numpy as np
from scipy.sparse import coo_matrix

from utils import mesh


def lk():
    E = 1
    nu = 0.3
    k = np.array(
        [1 / 2 - nu / 6, 1 / 8 + nu / 8, -1 / 4 - nu / 12, -1 / 8 + 3 * nu / 8, -1 / 4 + nu / 12, -1 / 8 - nu / 8,
         nu / 6, 1 / 8 - 3 * nu / 8])
    KE = E / (1 - nu ** 2) * np.array([[k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                                       [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                                       [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                                       [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                                       [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                                       [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                                       [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                                       [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]]);
    return KE


n_elx, n_ely = 10, 10
nodes, elements = mesh(1, 1, n_elx, n_ely, {})

e_dof_mat = np.zeros((n_elx * n_ely, 8), dtype=int)
for ii in range(n_elx):
    for jj in range(n_ely):
        el = jj + ii * n_ely
        nodes = elements[ii, jj].nodes
        e_dof_mat[el, :] = 2 * (np.repeat(nodes, 2) - 1) + np.repeat([[0, 1]], 4, axis=0).flatten()
iK = np.kron(e_dof_mat, np.ones((8, 1))).flatten()
jK = np.kron(e_dof_mat, np.ones((1, 8))).flatten()

KE = lk()
x = np.ones(n_ely * n_elx, dtype=float)
xx = np.ones(elements.shape)

local_k = np.empty(elements.shape, dtype=object)
for i in range(elements.shape[1]):
    for j in range(elements.shape[0]):
        local_k[i, j] = (xx[i, j] ** 3) * KE

sK = ((KE.flatten()[np.newaxis]).T * (x ** 3)).flatten(order='F')
K1 = coo_matrix((sK, (iK, jK)), shape=(2 * (n_elx + 1) * (n_ely + 1), 2 * (n_elx + 1) * (n_ely + 1))).tocsc()
K2 = assemble2(elements, local_k, 2 * (n_elx + 1) * (n_ely + 1))
print(np.allclose(K1.toarray(), K2.toarray()))
pass
