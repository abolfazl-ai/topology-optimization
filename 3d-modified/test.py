import numpy as np
from scipy.optimize import minimize


def rosen_with_args(x, a, b):
    """The Rosenbrock function with additional arguments"""
    return 1


x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
res = minimize(rosen_with_args, x0, method='nelder-mead',
               args=(0.5, 1.), options={'xatol': 1e-8, 'disp': True})
print(res.x)
