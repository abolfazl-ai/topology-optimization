import numpy as np
import pandas as pd
import pyvista as pv

p = pv.Plotter(shape=(1, 2))
actor = None


def plot_3d(x, m_d, m_c):
    global p, actor
    thresh = 0.5

    array = x.copy()
    for i, d in enumerate(m_d):
        array[(x > ((m_d[i - 1] + d) / 2 if i > 0 else d))] = i

    grid = pv.ImageData(dimensions=x.shape, spacing=(1, 1, 1), origin=(0, 0, 0))
    grid.point_data['Density'] = array.flatten(order="F")
    roi = grid.gaussian_smooth().threshold(thresh, invert=False)

    p.subplot(0, 0)
    p.add_mesh(roi, style='wireframe', color='gray', opacity=0.05)
    # actor = p.add_mesh(roi, cmap=m_c)
    #
    # def my_plane_func(normal, origin):
    #     global actor
    #     print(origin)
    #     p.remove_actor(actor)
    #     clip = roi.clip(normal=normal, origin=origin)
    #     actor = p.add_mesh(clip, cmap=m_c)
    #
    # p.add_plane_widget(my_plane_func, normal='y', origin=(np.array(x.shape) - 1) * (0.5, 1, 0.5), factor=1)
    p.add_mesh_clip_plane(roi, normal='y', origin=(np.array(x.shape) - 1) * (0.5, 1, 0.5),
                          invert=True, value=1.0, cmap=m_c)
    p.subplot(0, 1)
    p.add_mesh(roi, cmap=m_c)
    p.link_views()
    p.show(full_screen=True)


df = pd.read_excel('xs.xlsx', engine='openpyxl')
x = np.reshape(df.values, (30, 30, 30), 'F')
plot_3d(x, [0.5, 0.75, 1], ['red', 'blue', 'green'])
