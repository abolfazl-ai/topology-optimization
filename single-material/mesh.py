import numpy as np
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
