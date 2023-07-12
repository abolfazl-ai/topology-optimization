import numpy as np
import matplotlib.pyplot as plt

from NewCode.plot_results import plot_output
from boundaries import apply_bc, create_load_bc
from mesh import mesh
from stiffness import q4_k_matrix, assemble


def main(length, width, n_elx, n_ely, vol_frac, penalty, r_min):
    bc = create_load_bc(length, width, m=n_elx)  # Creating load boundary condition
    nodes, elements = mesh(length, width, n_elx, n_ely, bc)
    dof = {num: len(single_node.displacement) for num, single_node in nodes.items()}
    diff = {m: sum(dof[i] for i in range(1, m)) for m in nodes}

    KE = q4_k_matrix(elements[0, 0], nodes)
    x = vol_frac * np.ones(elements.shape)

    loop = 0
    change = 1
    while change > 0.01:
        loop += 1
        x_old = x.copy()
        U = fem(nodes, elements, KE, x, penalty, dof, diff)
        c, dc = 0.0, np.zeros(elements.shape)
        for i in range(elements.shape[1]):
            for j in range(elements.shape[0]):
                Ue = elements[i, j].get_primary_variables(U, diff)
                c += (x[i, j] ** penalty) * ((Ue.T @ KE) @ Ue)
                dc[i, j] = -penalty * (x[i, j] ** (penalty - 1)) * ((Ue.T @ KE) @ Ue)

        x = oc(n_elx, n_ely, vol_frac, x, dc)
        change = np.max(abs(x - x_old))
        plt.imshow(-x.T, cmap='gray', vmin=-1, vmax=0, origin='lower')

        obj = (x ** penalty).sum()
        print(f"it.: {loop} , obj.: {obj} , ch.: {change}")
        if change > 0.01:
            plt.pause(1E-6)
        else:
            plt.show()


def fem(nodes, elements, KE, x, penalty, dof, diff):
    U, F = np.zeros(sum(dof.values())), np.zeros(sum(dof.values()))
    for node in nodes:
        U[diff[node]:diff[node] + dof[node]] = nodes[node].displacement
        F[diff[node]:diff[node] + dof[node]] = nodes[node].force

    local_k = np.empty(elements.shape, dtype=object)
    for i in range(elements.shape[1]):
        for j in range(elements.shape[0]):
            local_k[i, j] = (x[i, j] ** penalty) * KE

    K = assemble(elements, local_k, dof)  # Assembling the global stiffness matrix
    U = apply_bc(K, U, F)  # Applying boundary conditions
    # plot_output(nodes, elements, U)
    return U


def oc(n_elx, n_ely, vol_frac, x, dc):
    l1, l2, move = 0, 100000, 0.2
    x_new = np.zeros(x.shape)
    while (l2 - l1) / (l1 + l2) > 1e-3:
        l_mid = 0.5 * (l2 + l1)
        x_new[:] = np.maximum(0.001, np.maximum(x - move,
                                                np.minimum(1.0, np.minimum(x + move, x * np.sqrt(-dc / l_mid)))))
        if np.concatenate(x_new).sum() - vol_frac * n_elx * n_ely > 0:
            l1 = l_mid
        else:
            l2 = l_mid
    return x_new


main(1, 1, 50, 50, 0.3, 3, 1)
