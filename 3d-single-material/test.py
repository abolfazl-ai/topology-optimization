import numpy as np
from scipy.sparse import csc_matrix, dok_matrix

from plot import plot_3d
import pandas as pd

plot_3d('x.npy', [0.5, 0.75, 1], ['Material A', 'Material B', 'Material C'], ['r', 'b', 'g'])

