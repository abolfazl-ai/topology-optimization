import numpy as np
import pyvista as pv

p = pv.Plotter(shape=(1, 2))
threshold = 0.5


def plot_3d(path, densities, names, colors):
    x = np.load(path)
    global threshold
    array = x.copy()
    for i, d in enumerate(densities):
        array[(x > ((densities[i - 1] + d) / 2 if i > 0 else d))] = i

    grid = pv.ImageData(dimensions=x.shape, spacing=(1, 1, 1), origin=(0, 0, 0))
    init_volume = grid.volume
    grid.point_data['Density'] = x.flatten(order="F")
    grid.point_data['Color'] = array.flatten(order="F")
    grid = grid.gaussian_smooth(scalars='Color')
    roi = grid.clip_scalar(value=threshold, scalars='Density', invert=False)

    def set_threshold(value):
        global threshold
        threshold = value
        p.remove_actor(('isosurface', 'volume'))
        clipped = grid.clip_scalar(value=threshold, scalars='Density', invert=False)
        p.add_mesh(clipped, scalars='Color', cmap=colors, show_scalar_bar=False, name='isosurface')
        p.add_text(f'Volume = {100 * clipped.volume / init_volume:0.1f}%', 'lower_left', 12, name='volume')

    p.subplot(0, 0)
    p.add_axes()
    p.add_legend(list(zip(names, colors)), bcolor='w', loc='upper left', face=None, size=(0.15, 0.18))
    p.add_mesh(roi, style='wireframe', color='gray', opacity=0.05, name='wireframe')
    p.add_mesh_clip_plane(roi, normal='x', origin=(np.array(x.shape) + 1) ** 2, assign_to_axis='x',
                          invert=True, scalars='Color', cmap=colors, show_scalar_bar=False, name='clip')
    p.subplot(0, 1)
    p.add_slider_widget(set_threshold, [0, 1], threshold, 'Threshold', (0.1, 0.9), style='modern', fmt='%0.2f')
    p.add_bounding_box()
    p.link_views()
    p.show(full_screen=False, window_size=(1000, 500))
