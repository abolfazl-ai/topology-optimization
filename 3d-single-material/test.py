import numpy as np
import pyvista as pv

p = pv.Plotter(shape=(1, 2))
threshold = 0.3


def plot_3d(path, densities, names, colors):
    x = np.load(path)
    global threshold
    array = x.copy()
    for i, d in enumerate(densities):
        array[(x > ((densities[i - 1] + d) / 2 if i > 0 else d))] = i

    grid = pv.ImageData(dimensions=x.shape, spacing=(1, 1, 1), origin=(0, 0, 0))
    grid.point_data['Density'] = x.flatten(order="F")
    grid.point_data['Color'] = array.flatten(order="F")
    roi = grid.clip_scalar(value=threshold, scalars='Density', invert=False)

    def set_threshold(value):
        global threshold
        threshold = value
        p.remove_actor('isosurface')
        p.add_mesh(grid.clip_scalar(value=threshold, scalars='Density', invert=False),
                   scalars='Color', cmap=colors, name='isosurface')

    p.subplot(0, 0)
    p.add_axes()
    p.add_legend(list(zip(names, colors)), bcolor='w', face=None, size=(0.15, 0.18))
    p.add_mesh(roi, style='wireframe', color='gray', opacity=0.05, name='wireframe')
    p.add_mesh_clip_plane(roi, normal='y', origin=(np.array(x.shape) + 1) ** 2, assign_to_axis='y',
                          invert=True, scalars='Color', cmap=colors, show_scalar_bar=False, name='clip')
    p.subplot(0, 1)
    p.add_slider_widget(set_threshold, [0, 1], threshold, pointa=(0.1, 0.9), pointb=(0.9, 0.9), title='Threshold')
    p.add_bounding_box()
    p.link_views()
    p.show(full_screen=False, window_size=(1000, 500))


plot_3d('output/it194.npy',
        [0.5, 0.75, 1],
        ['Material A', 'Material B', 'Material C'],
        ['red', 'blue', 'green'])
