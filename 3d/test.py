import numpy as np
from plot import Plot3D
path = '45D2.00.npy'
p = Plot3D(path, [0, 1], 'V,A', ['w', 'b'])
p.update(path, interactive=False)
