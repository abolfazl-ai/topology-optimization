import copy
import scipy.sparse.linalg as spla
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from classes import Node, Element


def mesh(length, width, m, n, bc, E=1, v=0.3, t=0.02, stress_mode=0):
    nodes = dict()
    for i in range(m + 1):
        for j in range(n + 1):
            num = i * (n + 1) + j + 1
            position = ((length / m) * i, (width / n) * j)
            displacement, force = (np.nan, np.nan), (0, 0)
            if position in bc:
                displacement = bc[position][0]
                force = bc[position][1]
            nodes[num] = Node(num, position, force, displacement)

    elements = np.empty(shape=(m, n), dtype=object)
    for i in range(m):
        for j in range(n):
            elem_num = i * n + j + 1
            n1 = i * (n + 1) + j + 1
            n2 = (i + 1) * (n + 1) + j + 1
            n3 = (i + 1) * (n + 1) + j + 2
            n4 = i * (n + 1) + j + 2
            elements[i, j] = Element(elem_num, (n1, n2, n3, n4), E, v, t, stress_mode)

    return nodes, elements


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


def plot_output(nodes, elements, d, exaggeration=0):
    if exaggeration == 0:
        exaggeration = 0.1 / max(abs(d))

    fig, ax = plt.subplots()

    new_nodes = copy.deepcopy(nodes)
    for n, node in new_nodes.items():
        new_position = (node.position[0] + exaggeration * d[int(2 * (n - 1))],
                        node.position[1] + exaggeration * d[int(2 * (n - 1) + 1)])
        node.position = new_position

    for i in range(elements.shape[1]):
        for j in range(elements.shape[0]):
            new_node_positions = [new_nodes[node].position for node in elements[i, j].nodes]
            old_node_positions = [nodes[node].position for node in elements[i, j].nodes]
            ax.plot(*zip(*old_node_positions, old_node_positions[0]), color='blue', zorder=1)
            ax.plot(*zip(*new_node_positions, new_node_positions[0]), color='black', zorder=1)

    ax.scatter(*zip(*[node.position for node in nodes.values()]), marker='o', linewidths=0.5, color='blue', zorder=2)
    ax.scatter(*zip(*[node.position for node in new_nodes.values()]), marker='o', linewidths=0.5, color='red', zorder=2)

    for node in nodes.values():
        ax.annotate(node.number, node.position, textcoords="offset points", xytext=(-6, -8), ha='center', fontsize=6)

    ax.set(xlabel="X", ylabel="Y", title='Elements after deformation', aspect=1,
           xlim=(ax.get_xlim()[0] - 1, ax.get_xlim()[1] + 1), ylim=(ax.get_ylim()[0] - 0.5, ax.get_ylim()[1] + 0.5))

    plt.show()
