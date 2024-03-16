import numpy as np
from cvxopt import matrix
from inputs import *


def get_input():
    mesh = mesh_model()
    bc = gen_bc(mesh, boundary_conditions)
    pres, mask = read_pres(mesh)
    return mesh, bc, filter_parameters, optimization_parameters, materials, pres, mask


def mesh_model():
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


def read_pres(mesh):
    pres, mask = np.zeros(mesh['shape']), np.ones(mesh['shape'], dtype=bool)
    for row in preserved_regions:
        start, end = ([float(x) for x in row['S'].split(',')],
                      [float(x) for x in row['E'].split(',')])
        nr = [(int(max(min(np.floor(s * n), np.floor(e * n) - 1), 0)), int(np.floor(e * n)) + 1)
              for n, s, e in list(zip(mesh['shape'], start, end))]
        if mesh['dim'] == 3:
            mask[nr[1][0]:nr[1][1], nr[2][0]:nr[2][1], nr[0][0]:nr[0][1]] = False
            pres[nr[1][0]:nr[1][1], nr[2][0]:nr[2][1], nr[0][0]:nr[0][1]] = row['D']
        else:
            mask[nr[1][0]:nr[1][1], nr[0][0]:nr[0][1]] = False
            pres[nr[1][0]:nr[1][1], nr[0][0]:nr[0][1]] = row['D']
    return pres, mask


def gen_bc(mesh, bc):
    fixed, dof = [], mesh['dim'] * np.prod(np.array(mesh['shape']) + 1)
    force_vector = matrix(0.0, (dof, 1))
    for row in bc:
        start, end, displacement, force = [row[key] for key in ('S', 'E', 'D', 'F')]
        nr = [(int(np.floor(s * n)), int(np.floor(e * n)) + 1) for n, s, e in list(zip(mesh['shape'], start, end))]
        nodes = (mesh['node_numbers'][nr[1][0]:nr[1][1], nr[0][0]:nr[0][1]] if mesh['dim'] == 2 else
                 mesh['node_numbers'][nr[1][0]:nr[1][1], nr[2][0]:nr[2][1], nr[0][0]:nr[0][1]]).flatten()
        for i in range(len(mesh['shape'])):
            if force != 0:
                for node in nodes:
                    coordinates = np.argwhere(mesh['node_numbers'] == node)[0] / mesh['shape']
                    force_vector[(mesh['dim'] * node + i).tolist(), 0] = force(*coordinates)[i]
            fixed.extend([] if np.isnan(displacement[i]) else (mesh['dim'] * nodes + i).tolist())
    free = np.setdiff1d(np.arange(0, dof), fixed).tolist()
    return {'free_dofs': free, 'force_vector': force_vector}
