import matplotlib.pyplot as plt
from modes import *
import numpy as np
import os
import tensorflow as tf

# def load_data(bc_code, load_code, penal, r, vf):
#     bc_code, load_code, penal, r, vf = [param.numpy() for param in (bc_code, load_code, penal, r, vf)]
#     bc_code, load_code = bc_code.decode('utf-8'), load_code.decode('utf-8')
#     load_path = 'data/' + bc_code + '/' + load_code + '/'
#     code = f'{bc_code}-{load_code}-P{penal}-R{r:0.3f}-V{vf:0.3f}'
#
#     c = np.load(load_path + bc_code + load_code + '-C.npy')
#     ux = np.load(load_path + bc_code + load_code + '-X.npy')
#     uy = np.load(load_path + bc_code + load_code + '-Y.npy')
#
#     input_2d = np.ones((201, 201, 4), dtype=np.float32)
#     input_2d[:, :, 0] = ux
#     input_2d[:, :, 1] = uy
#     input_2d[:, :, 2] = np.hypot(ux, uy)
#     input_2d[:, :, 3] = np.pad(c, pad_width=1, mode='edge')[0:-1, 0:-1]
#
#     y = np.load(load_path + 'output/' + code + '.npy').astype(np.float32)
#
#     return np.array((penal, r, vf)), input_2d, y
#
#
# def general_function(bc_code, load_code, penal, r, vf):
#     input_1d, input_2d, y = tf.py_function(load_data, (bc_code, load_code, penal, r, vf),
#                                            (tf.float32, tf.float32, tf.float32))
#     input_1d.set_shape(tf.TensorShape([3, ]))
#     input_2d.set_shape(tf.TensorShape([201, 201, 4]))
#     y.set_shape(tf.TensorShape([200, 200]))
#     return (input_1d, input_2d), y
#
#
# parameters = [(b, l, p, r, v) for b in list(bcs.keys()) for l in list(loads.keys())
#               for p in ps for r in rs for v in vfs]
# np.random.shuffle(parameters)
# parameters = tuple(np.array(item) for item in zip(*parameters))
#
#
# def filter_function(bc_code, load_code, penal, r, vf):
#     bc_code, load_code, penal, r, vf = [param.numpy() for param in (bc_code, load_code, penal, r, vf)]
#     bc_code, load_code = bc_code.decode('utf-8'), load_code.decode('utf-8')
#     load_path = 'data/' + bc_code + '/' + load_code + '/'
#     code = f'{bc_code}-{load_code}-P{penal}-R{r:0.3f}-V{vf:0.3f}'
#     y_path = load_path + 'output/' + code + '.npy'
#     c_path = load_path + bc_code + load_code + '-C.npy'
#     return os.path.exists(y_path) and np.load(c_path).any()
#
#
# dataset = tf.data.Dataset.from_tensor_slices(parameters)
# dataset = dataset.filter(lambda b, l, p, r, v: tf.py_function(filter_function, (b, l, p, r, v), tf.bool))
# dataset = dataset.map(lambda b, l, p, r, v: general_function(b, l, p, r, v))
# dataset = dataset.batch(25)
#
# batch = next(dataset.as_numpy_iterator())
# x = batch[0][1][0]
# c = x[:, :, 3]
# y = batch[1][0]
# np.save(arr=y, file='YYY')
# np.save(arr=c, file='CCC')

import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
c = np.load('CCC.npy')
c = MinMaxScaler().fit_transform(c)
y = np.load('YYY.npy')
fig, ax = plt.subplots(1, 2)
sns.heatmap(c, ax=ax[0], robust=False)
sns.heatmap(y, ax=ax[1])
plt.show()
