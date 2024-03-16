import numpy as np

# Mesh parameters
nx, ny, nz = 400, 400, 0

# Optimization parameters
optimization_parameters = {
    'penalty': 3, 'penalty_increase': 0.1, 'volume_fraction': 0.2, 'move': 0.2,
    # Convergence options (Maximum change in o_conv: objective and x_conv: density)
    'max_it': 200, 'o_conv': 0.0001, 'x_conv': 0.001}

# Filter parameters
filter_parameters = {
    'filter': 3,  # 1: Only density | 2 and 3: Projection with eta and beta
    'filter_bc': ['constant', 'reflect', 'nearest', 'mirror', 'wrap'][1],
    'radius': 2, 'eta': 0.2, 'beta': 2}

# Boundary conditions and loads
# S: Starting point, E: End point, D: Displacement, F: Force
boundary_conditions = [
    {'S': (0, 0, 0), 'E': (1, 0, 0), 'D': (0, 0, 0), 'F': 0},
    {'S': (0, 1, 0), 'E': (1, 1, 0), 'D': (np.nan, np.nan, np.nan), 'F': lambda x, y, z=0: (0, -1, 0)}]

# Non-design regions
# S: Starting point, E: End point, D: Fixed value
preserved_regions = [
    # {'S': (0, 0, 0), 'E': (1, 1, 0), 'D': (0, 0, 0)},
]

# Materials
materials = {'names': ['Void', 'Solid'],  # Display name of material
             'colors': ['w', 'k'],  # Display color of material
             'D': [1e-9, 1],  # Normalized material density
             'E': [1e-9, 1]}  # Normalized material elastic modulus
