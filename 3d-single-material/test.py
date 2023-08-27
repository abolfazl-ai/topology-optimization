import numpy as np
import pyvista as pv

p = pv.Plotter(shape=(1, 2))


def plot_3d(x, densities, names, colors):
    array = x.copy()
    for i, d in enumerate(densities):
        array[(x > ((densities[i - 1] + d) / 2 if i > 0 else d))] = i

    grid = pv.ImageData(dimensions=x.shape, spacing=x.shape, origin=(0, 0, 0))
    grid.point_data['Color'] = array.flatten(order="F")
    grid.point_data['Density'] = x.flatten(order="F")

    roi = grid.gaussian_smooth().threshold(0.5, invert=False)

    p.subplot(0, 0)
    p.add_mesh(roi, style='wireframe', color='gray', opacity=0.05)
    p.add_mesh_clip_plane(roi, normal='y', origin=(np.array(x.shape) - 1) * (0.5, 1, 0.5), invert=True, value=1.0,
                          assign_to_axis='y', scalars='Color', cmap=colors, show_scalar_bar=False)
    p.add_legend(list(zip(names, colors)), bcolor='w', loc='upper left', face=None, size=(0.15, 0.18))
    p.subplot(0, 1)
    p.add_mesh_threshold(grid, scalars='Density', title='Threshold', cmap=['gray'], show_scalar_bar=False,
                         pointa=(0.1, 0.9), pointb=(0.9, 0.9))
    p.add_bounding_box()
    p.link_views()
    p.show(full_screen=False, window_size=(1000, 500))


x = np.load('x.npy')
plot_3d(x, [0.5, 0.75, 1], ['Material A', 'Material B', 'Material C'], ['red', 'blue', 'green'])
