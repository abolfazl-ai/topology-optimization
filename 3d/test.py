import numpy as np
from plot import Plot3D

path = 'runs/45D1.73.npy'
p = Plot3D(path, [0, 1], ['V', 'A'], ['w', 'b'])
p.update(path, interactive=False)
