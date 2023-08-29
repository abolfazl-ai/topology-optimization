import multiprocessing
import threading
import time

from plot import Plot3D

p = Plot3D('final.npy', [0.5, 0.75, 1],
           ['Material A', 'Material B', 'Material C'],
           ['r', 'b', 'g'], interactive=True)

time.sleep(5)
s = time.time()
p.update('x.npy', interactive=True)
print(time.time() - s)
time.sleep(5)
p.update('final.npy', interactive=False)
