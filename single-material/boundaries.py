import numpy as np
import scipy.sparse.linalg as spla


def create_load_bc(length, width, m):
    # x_values = np.linspace(0, length, m + 1)
    # fix_y_value, load_y_values = 0, width
    # bc = {(x, fix_y_value): ((0, 0), (np.nan, np.nan)) for x in x_values}
    # load = {(x, load_y_values): ((np.nan, np.nan), (0, -1 / (m + 1))) for x in x_values}
    bc = {(0, 0): ((0, 0), (np.nan, np.nan)), (1, 0): ((0, 0), (np.nan, np.nan))}
    load = {(0.2, 1): ((np.nan, np.nan), (0, -1)), (0.8, 1): ((np.nan, np.nan), (0, -1))}
    return bc | load


def apply_bc(stiffness, displacement, forces):
    U = np.copy(displacement)
    free = np.where(np.isnan(displacement))[0]
    K = stiffness[free, :][:, free].tocsr()
    U[free] = spla.spsolve(K, forces[free])
    return U
