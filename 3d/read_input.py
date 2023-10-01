import numpy as np
import pandas as pd
from cvxopt import matrix


def read_options(input_path):
    options = pd.read_excel(input_path, sheet_name='Options')
    nx, ny, nz, vf, penalty, max_it, move, ft, filter_bc, r_min, eta, beta = options['Value']
    nx, ny, nz, penalty, ft, filter_bc = np.array((nx, ny, nz, penalty, ft, filter_bc), dtype=np.int32)
    filter_bc = ['constant', 'reflect', 'nearest', 'mirror', 'wrap'][filter_bc]
    return nx, ny, nz, vf, penalty, ft, filter_bc, max_it, r_min, eta, beta, move


def read_materials(input_path):
    material_df = pd.read_excel(input_path, sheet_name='Materials')
    D = material_df['Density'].tolist()
    E = material_df['Elasticity'].tolist()
    D[D == 0] = 1e-9
    E[E == 0] = 1e-9
    M_name = material_df['Material'].tolist()
    M_color = material_df['Color'].tolist()
    return D, E, M_name, M_color


def read_bc(input_path, nx, ny, nz, node_numbers):
    bc = pd.read_excel(input_path, sheet_name='BC')
    fixed, dof = [], 3 * (1 + nx) * (1 + ny) * (1 + nz)
    force_vector = matrix(0.0, (dof, 1))
    for _, row in bc.iterrows():
        start, end = ([float(x) for x in row['StartPosition'].split(',')],
                      [float(x) for x in row['EndPosition'].split(',')])
        displacement, force = [row['DisX'], row['DisY'], row['DisZ']], [row['ForceX'], row['ForceY'], row['ForceZ']]
        nr = [(int(np.floor(s * n)), int(np.floor(e * n)) + 1) for n, s, e in list(zip((nx, ny, nz), start, end))]
        nodes = node_numbers[nr[1][0]:nr[1][1], nr[2][0]:nr[2][1], nr[0][0]:nr[0][1]].flatten()
        for i, (d, f) in enumerate(zip(displacement, force)):
            force_vector[(3 * nodes + i).tolist(), 0] = matrix(0 if np.isnan(f) else f, (nodes.size, 1))
            fixed.extend([] if np.isnan(d) else (3 * nodes + i).tolist())
    free = np.setdiff1d(np.arange(0, dof), fixed).tolist()
    return free, force_vector


def read_pres(input_path, nx, ny, nz):
    preserved = pd.read_excel(input_path, sheet_name='PreservedVolume')
    pres, mask = np.zeros((ny, nz, nx)), np.ones((ny, nz, nx), dtype=bool)
    for _, row in preserved.iterrows():
        start, end = ([float(x) for x in row['StartPosition'].split(',')],
                      [float(x) for x in row['EndPosition'].split(',')])
        nr = [(int(max(min(np.floor(s * n), np.floor(e * n) - 1), 0)), int(np.floor(e * n)) + 1)
              for n, s, e in list(zip((nx, ny, nz), start, end))]
        mask[nr[1][0]:nr[1][1], nr[2][0]:nr[2][1], nr[0][0]:nr[0][1]] = False
        pres[nr[1][0]:nr[1][1], nr[2][0]:nr[2][1], nr[0][0]:nr[0][1]] = row['Density']
    return pres, mask
