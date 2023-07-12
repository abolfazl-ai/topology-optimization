import numpy as np
import pandas as pd
import scipy.sparse.linalg as spla


def create_load_bc(m, n, input_path='bc-load.xlsx'):
    bc_df = pd.read_excel(input_path, sheet_name='BC')
    bc = dict()
    for index, row in bc_df.iterrows():
        sx, sy, ex, ey = row['StartX'], row['StartY'], row['EndX'], row['EndY']
        if sx == ex and sy == ey:
            bc[(sx, sy)] = ((row['DisplacementX'], row['DisplacementY']), (row['ForceX'], row['ForceY']))
        else:
            for x in np.linspace(sx, ex, int(max(1, (ex - sx) * (m + 1)))):
                for y in np.linspace(sy, ey, int(max(1, (ey - sy) * (n + 1)))):
                    bc[(x, y)] = ((row['DisplacementX'], row['DisplacementY']), (row['ForceX'], row['ForceY']))
    return bc


def apply_bc(stiffness, displacement, forces):
    U = np.copy(displacement)
    free = np.where(np.isnan(displacement))[0]
    K = stiffness[free, :][:, free].tocsr()
    U[free] = spla.spsolve(K, forces[free])
    return U
