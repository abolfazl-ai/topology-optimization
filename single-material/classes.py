import numpy as np


# Utility class for nodes
class Node:
    def __init__(self, number, position, force, displacement):
        self.number = number
        self.position = position
        self.force = force
        self.displacement = displacement

    def get_primary_variables(self, d, diff):
        return np.array(d[diff[self.number]: diff[self.number] + len(self.displacement)])

    def get_secondary_variables(self, f, diff):
        return np.array(f[diff[self.number]: diff[self.number] + len(self.displacement)])


class Element:
    def __init__(self, number, nodes, young, poisson, thickness, stress_mode):
        self.number = number
        self.nodes = nodes
        self.young = young
        self.poisson = poisson
        self.thickness = thickness
        self.stress_mode = stress_mode

    def get_xy(self, all_nodes):
        X, Y = [], []
        for p in self.nodes:
            X.append(all_nodes[p].position[0])
            Y.append(all_nodes[p].position[1])
        return np.array(X).T, np.array(Y).T

    def get_primary_variables(self, d, diff):
        elem_d = []
        for elem_node in self.nodes:
            elem_d.append(d[diff[elem_node]])
            elem_d.append(d[diff[elem_node] + 1])
        return np.array(elem_d)
