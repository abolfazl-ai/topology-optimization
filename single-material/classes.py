# Utility class for nodes
class Node:
    def __init__(self, number, position, force, displacement):
        self.number = number
        self.position = position
        self.force = force
        self.displacement = displacement


class Element:
    def __init__(self, number, nodes, young, poisson, thickness, stress_mode):
        self.number = number
        self.nodes = nodes
        self.young = young
        self.poisson = poisson
        self.thickness = thickness
        self.stress_mode = stress_mode
