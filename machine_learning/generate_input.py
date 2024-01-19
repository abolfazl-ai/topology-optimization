import numpy as np
from cvxopt import matrix

from main.read_input import mesh_model


def generate_input(r, p, vol_fraction, boundary):
    materials = {'names': ['Void', 'Solid'], 'colors': ['w', 'k'], 'D': [1e-9, 1], 'E': [1e-9, 1]}
    mesh = mesh_model(400, 400, 0)
    filter_params = {'filter': 3, 'filter_bc': 'reflect', 'radius': r, 'eta': 0.2, 'beta': 2}
    opt_params = {'penalty': p, 'penalty_increase': 0.0, 'volume_fraction': vol_fraction, 'move': 0.2,
                  'max_it': 200, 'c_conv': 0.0001, 'x_conv': 0.001}
    pres, mask = np.zeros(mesh['shape']), np.ones(mesh['shape'], dtype=bool)

    bc = gen_bc(mesh, boundary)
    return mesh, bc, filter_params, opt_params, materials, pres, mask


def gen_bc(mesh, bc):
    fixed, dof = [], mesh['dim'] * np.prod(np.array(mesh['shape']) + 1)
    force_vector = matrix(0.0, (dof, 1))
    for row in bc:
        start, end, displacement, force = [row[key] for key in ('S', 'E', 'D', 'F')]
        nr = [(int(np.floor(s * n)), int(np.floor(e * n)) + 1) for n, s, e in list(zip(mesh['shape'], start, end))]
        nodes = (mesh['node_numbers'][nr[1][0]:nr[1][1], nr[0][0]:nr[0][1]]).flatten()
        for i, d in enumerate(displacement):
            if force != 0:
                for node in nodes:
                    y, x = np.argwhere(mesh['node_numbers'] == node)[0] / mesh['shape']
                    force_vector[(mesh['dim'] * node + i).tolist(), 0] = force(x, y)[i]
            fixed.extend([] if np.isnan(d) else (mesh['dim'] * nodes + i).tolist())
    free = np.setdiff1d(np.arange(0, dof), fixed).tolist()
    return {'free_dofs': free, 'force_vector': force_vector}
