import numpy as np
import pandas as pd
from cvxopt import matrix


def get_input(input_path):
    mesh, filter_params, opt_params = read_options(input_path)
    pres, mask = read_pres(input_path, mesh)
    bc = read_bc(input_path, mesh)
    materials = read_materials(input_path)
    return mesh, bc, filter_params, opt_params, materials, pres, mask


def read_options(input_path):
    options = pd.read_excel(input_path, sheet_name='Options')
    nx, ny, nz, vf, penalty, max_it, x_conv, c_conv, move, ft, filter_bc, r, eta, beta = options['Value']
    nx, ny, nz, penalty, max_it, ft, filter_bc = np.array((nx, ny, nz, penalty, max_it, ft, filter_bc), np.int32)
    filter_bc = ['constant', 'reflect', 'nearest', 'mirror', 'wrap'][filter_bc]
    mesh = mesh_model(nx, ny, nz)
    filter_params = {'filter': ft, 'filter_bc': filter_bc, 'radius': r, 'eta': eta, 'beta': beta}
    opt_params = {'penalty': penalty, 'volume_fraction': vf, 'move': move,
                  'max_it': max_it, 'c_conv': c_conv, 'x_conv': x_conv}
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


def mesh_model(nx, ny, nz):
    shape, dim = ((nx, ny), 2) if nz == 0 else ((nx, ny, nz), 3)
    node_numbers = np.reshape(range(np.prod(np.array(shape) + 1)), np.array(shape) + 1, order='F')
    elem_num, dof = np.prod(shape), np.prod(np.array(shape) + 1) * dim
    if dim == 3:
        c_vec = np.reshape(dim * node_numbers[0: -1, 0: -1, 0: -1] + dim, (np.prod(shape), 1), order='F')
        additions = [np.array([0, 1, 2], dtype=np.int32),
                     np.array([3 * (ny + 1) * (nz + 1) + i for i in [0, 1, 2, -3, -2, -1]], dtype=np.int32),
                     np.array([-3, -2, -1], dtype=np.int32),
                     np.array([3 * (ny + 1) + i for i in [0, 1, 2]], dtype=np.int32),
                     np.array([3 * (ny + 1) * (nz + 2) + i for i in [0, 1, 2, -3, -2, -1]], dtype=np.int32),
                     np.array([3 * (ny + 1) + i for i in [-3, -2, -1]], dtype=np.int32)]
        c_mat = c_vec + np.concatenate(additions)
        sI, sII = np.hstack([np.arange(j, 24) for j in range(24)]), np.hstack([np.tile(j, 24 - j) for j in range(24)])
    else:
        c_vec = np.reshape(2 * node_numbers[0: -1, 0: -1] + 2, (nx * ny, 1), order='F')
        c_mat = c_vec + np.array([0, 1, 2 * ny + 2, 2 * ny + 3, 2 * ny + 0, 2 * ny + 1, -2, -1])
        sI, sII = np.hstack([np.arange(j, 8) for j in range(8)]), np.hstack([np.tile(j, 7 - j + 1) for j in range(8)])

    iK, jK = c_mat[:, sI].T, c_mat[:, sII].T
    indexes = np.sort(np.hstack((iK.reshape((-1, 1), order='F'), jK.reshape((-1, 1), order='F'))))[:, [1, 0]]
    return {'dim': dim, 'shape': shape, 'dof': dof, 'node_numbers': node_numbers,
            'elem_num': elem_num, 'indexes': (indexes[:, 0], indexes[:, 1]), 'c_mat': c_mat}


def read_bc(input_path, mesh):
    bc = pd.read_excel(input_path, sheet_name='BC')
    fixed, dof = [], mesh['dim'] * np.prod(np.array(mesh['shape']) + 1)
    force_vector = matrix(0.0, (dof, 1))
    for _, row in bc.iterrows():
        start, end = [[float(x) for x in row[s].split(',')][0:mesh['dim']] for s in ('StartPosition', 'EndPosition')]
        displacement, force = [[row[s + ('X', 'Y', 'Z')[i]] for i in range(mesh['dim'])] for s in ('Dis', 'Force')]
        nr = [(int(np.floor(s * n)), int(np.floor(e * n)) + 1) for n, s, e in list(zip(mesh['shape'], start, end))]
        nodes = (mesh['node_numbers'][nr[1][0]:nr[1][1], nr[0][0]:nr[0][1]] if mesh['dim'] == 2 else
                 mesh['node_numbers'][nr[1][0]:nr[1][1], nr[2][0]:nr[2][1], nr[0][0]:nr[0][1]]).flatten()
        for i, (d, f) in enumerate(zip(displacement, force)):
            force_vector[(mesh['dim'] * nodes + i).tolist(), 0] = matrix(0 if np.isnan(f) else f, (nodes.size, 1))
            fixed.extend([] if np.isnan(d) else (mesh['dim'] * nodes + i).tolist())
    free = np.setdiff1d(np.arange(0, dof), fixed).tolist()
    return {'free_dofs': free, 'force_vector': force_vector}


def read_pres(input_path, mesh):
    preserved = pd.read_excel(input_path, sheet_name='PreservedVolume')
    pres, mask = np.zeros(mesh['shape']), np.ones(mesh['shape'], dtype=bool)
    for _, row in preserved.iterrows():
        start, end = ([float(x) for x in row['StartPosition'].split(',')],
                      [float(x) for x in row['EndPosition'].split(',')])
        nr = [(int(max(min(np.floor(s * n), np.floor(e * n) - 1), 0)), int(np.floor(e * n)) + 1)
              for n, s, e in list(zip(mesh['shape'], start, end))]
        if mesh['dim'] == 3:
            mask[nr[1][0]:nr[1][1], nr[2][0]:nr[2][1], nr[0][0]:nr[0][1]] = False
            pres[nr[1][0]:nr[1][1], nr[2][0]:nr[2][1], nr[0][0]:nr[0][1]] = row['Density']
        else:
            mask[nr[1][0]:nr[1][1], nr[0][0]:nr[0][1]] = False
            pres[nr[1][0]:nr[1][1], nr[0][0]:nr[0][1]] = row['Density']
    return pres, mask
