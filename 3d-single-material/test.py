import numpy as np
import pandas as pd
import pyvista as pv

p = pv.Plotter(shape=(1, 2))


def plot_3d(x, densities, names, colors):
    thresh = 0.5

    array = x.copy()
    for i, d in enumerate(densities):
        array[(x > ((densities[i - 1] + d) / 2 if i > 0 else d))] = i

    grid = pv.ImageData(dimensions=x.shape, spacing=(1, 1, 1), origin=(0, 0, 0))
    grid.point_data['Density'] = array.flatten(order="F")
    roi = grid.gaussian_smooth().threshold(thresh, invert=False)

    p.subplot(0, 0)
    p.add_mesh(roi, style='wireframe', color='gray', opacity=0.05)
    p.add_mesh_clip_plane(roi, normal='y', origin=(np.array(x.shape) - 1) * (0.5, 1, 0.5), invert=True, value=1.0,
                          assign_to_axis='y', cmap=colors, show_scalar_bar=False)
    p.subplot(0, 1)
    p.add_mesh(roi, cmap=colors, show_scalar_bar=False)
    p.add_bounding_box()
    p.add_legend(list(zip(names, colors)), bcolor='w', face=None, size=(0.15, 0.18))
    p.link_views()
    p.show(full_screen=False, window_size=(1000, 500))


# df = pd.read_excel('xs.xlsx', engine='openpyxl')
# x = np.reshape(df.values, (30, 30, 30), 'F')
# plot_3d(x, [0.5, 0.75, 1], ['Material A', 'Material B', 'Material C'], ['red', 'blue', 'green'])
