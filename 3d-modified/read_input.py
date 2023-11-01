import numpy as np
import pandas as pd
from cvxopt import matrix


def mesh_model_3d(nx, ny, nz):
    node_numbers = np.reshape(range((1 + nx) * (1 + ny) * (1 + nz)), (1 + ny, 1 + nz, 1 + nx), order='F')
    elem_num, dof = nx * ny * nz, (1 + ny) * (1 + nx) * (1 + nz) * 3
    cVec = np.reshape(3 * node_numbers[0: -1, 0: -1, 0: -1] + 3, (nx * ny * nz, 1), order='F')
    additions = [np.array([0, 1, 2], dtype=np.int32),
                 np.array([3 * (ny + 1) * (nz + 1) + i for i in [0, 1, 2, -3, -2, -1]], dtype=np.int32),
                 np.array([-3, -2, -1], dtype=np.int32),
                 np.array([3 * (ny + 1) + i for i in [0, 1, 2]], dtype=np.int32),
                 np.array([3 * (ny + 1) * (nz + 2) + i for i in [0, 1, 2, -3, -2, -1]], dtype=np.int32),
                 np.array([3 * (ny + 1) + i for i in [-3, -2, -1]], dtype=np.int32)]
    cMat = cVec + np.concatenate(additions)
    sI, sII = np.hstack([np.arange(j, 24) for j in range(24)]), np.hstack([np.tile(j, 24 - j) for j in range(24)])
    iK, jK = cMat[:, sI].T, cMat[:, sII].T
    indexes = np.sort(np.hstack((iK.reshape((-1, 1), order='F'), jK.reshape((-1, 1), order='F'))))[:, [1, 0]]
    return {'shape': (nx, ny, nz), 'dof': dof, 'node_numbers': node_numbers, 'elem_num': elem_num,
            'indexes': (indexes[:, 0], indexes[:, 1]), 'c_mat': cMat}


def mesh_model_2d(nx, ny):
    node_numbers = np.reshape(range((1 + nx) * (1 + ny)), (1 + ny, 1 + nx), order='F')
    elem_num, dof = nx * ny, (1 + ny) * (1 + nx) * 2
    cVec = np.reshape(2 * node_numbers[0: -1, 0: -1] + 2, (nx * ny, 1), order='F')
    cMat = cVec + np.array([0, 1, 2 * ny + 2, 2 * ny + 3, 2 * ny + 0, 2 * ny + 1, -2, -1])
    sI, sII = np.hstack([np.arange(j, 8) for j in range(8)]), np.hstack([np.tile(j, 7 - j + 1) for j in range(8)])
    iK, jK = cMat[:, sI].T, cMat[:, sII].T
    indexes = np.sort(np.hstack((iK.reshape((-1, 1), order='F'), jK.reshape((-1, 1), order='F'))))[:, [1, 0]]
    return {'shape': (nx, ny), 'dof': dof, 'node_numbers': node_numbers, 'elem_num': elem_num,
            'indexes': (indexes[:, 0], indexes[:, 1]), 'c_mat': cMat}

def read_options(input_path):
    options = pd.read_excel(input_path, sheet_name='Options')
    nx, ny, nz, vf, penalty, max_it, x_converge, c_converge, move, ft, filter_bc, r, eta, beta = options['Value']
    nx, ny, nz, penalty, max_it, ft, filter_bc = np.array((nx, ny, nz, penalty, max_it, ft, filter_bc), dtype=np.int32)
    filter_bc = ['constant', 'reflect', 'nearest', 'mirror', 'wrap'][filter_bc]
    mesh = mesh_model_3d(nx, ny, nz)
    filter_params = {'filter': ft, 'filter_bc': filter_bc, 'radius': r, 'eta': eta, 'beta': beta}
    opt_params = {'penalty': penalty, 'volume_fraction': vf, 'move': move,
                  'max_iteration': max_it, 'c_conv': c_converge, 'x_conv': x_converge}
    return mesh, filter_params, opt_params


def read_materials(input_path):
    material_df = pd.read_excel(input_path, sheet_name='Materials')
    D = material_df['Density'].tolist()
    E = material_df['Elasticity'].tolist()
    D[D == 0] = 1e-9
    E[E == 0] = 1e-9
    names = material_df['Material'].tolist()
    colors = material_df['Color'].tolist()
    return {'names': names, 'colors': colors, 'D': D, 'E': E}


def read_bc(input_path, mesh):
    nx, ny, nz = mesh['shape']
    bc = pd.read_excel(input_path, sheet_name='BC')
    fixed, dof = [], 3 * (1 + nx) * (1 + ny) * (1 + nz)
    force_vector = matrix(0.0, (dof, 1))
    for _, row in bc.iterrows():
        start, end = ([float(x) for x in row['StartPosition'].split(',')],
                      [float(x) for x in row['EndPosition'].split(',')])
        displacement, force = [row['DisX'], row['DisY'], row['DisZ']], [row['ForceX'], row['ForceY'], row['ForceZ']]
        nr = [(int(np.floor(s * n)), int(np.floor(e * n)) + 1) for n, s, e in list(zip((nx, ny, nz), start, end))]
        nodes = mesh['node_numbers'][nr[1][0]:nr[1][1], nr[2][0]:nr[2][1], nr[0][0]:nr[0][1]].flatten()
        for i, (d, f) in enumerate(zip(displacement, force)):
            force_vector[(3 * nodes + i).tolist(), 0] = matrix(0 if np.isnan(f) else f, (nodes.size, 1))
            fixed.extend([] if np.isnan(d) else (3 * nodes + i).tolist())
    free = np.setdiff1d(np.arange(0, dof), fixed).tolist()
    return {'free_dofs': free, 'force_vector': force_vector}


def read_pres(input_path, mesh):
    nx, ny, nz = mesh['shape']
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
