import numpy as np
import pyvista as pv


class Plot3D:
    def __init__(self, path, densities_vector, names_vector, colors_vector, thresh=0.3):
        self.path, self.densities, self.names, self.colors = path, densities_vector, names_vector, colors_vector

        x, array = self._read_data(self.path)
        self.grid = pv.ImageData(dimensions=x.shape, spacing=(1, 1, 1), origin=(0, 0, 0))
        self.volume = self.grid.volume
        self.grid.point_data['Density'] = x.flatten(order="F")
        self.grid.point_data['Color'] = array.flatten(order="F")
        self.grid = self.grid.gaussian_smooth(scalars='Color')
        self.roi = self.grid.clip_scalar(value=thresh, scalars='Density', invert=False)

        self.p = pv.Plotter(shape=(1, 2))
        self.p.subplot(0, 0)
        self.p.add_legend(list(zip(self.names, self.colors)), 'w', loc='upper left', face=None, size=(0.15, 0.18))
        self.p.add_mesh(self.roi, style='wireframe', color='gray', opacity=0.05, name='wireframe')
        self.actor = self.p.add_mesh_clip_plane(self.roi, 'x', True, origin=np.array(x.shape) * 2, assign_to_axis='x',
                                                scalars='Color', cmap=self.colors, show_scalar_bar=False, name='clip')
        self.p.add_axes()
        self.p.subplot(0, 1)
        self.p.add_slider_widget(self._update, [0, 1], thresh, 'Threshold', (0.1, 0.9), style='modern', fmt='%0.2f')
        self.p.add_bounding_box()
        self.p.link_views()
        self.p.show(full_screen=False, window_size=(1000, 500))

    def _update(self, value):
        self.threshold = value
        self.clipped = self.grid.clip_scalar(value=self.threshold, scalars='Density', invert=False)
        self.p.subplot(0, 1)
        self.p.add_text(f'Volume = {100 * self.clipped.volume / self.volume:0.1f}%', (30, 30), 10, name='volume')
        self.p.add_mesh(self.clipped, scalars='Color', cmap=self.colors, show_scalar_bar=False, name='isosurface')

    def _read_data(self, path):
        x, array = np.load(path), np.load(path)
        for i, d in enumerate(self.densities):
            array[(x > ((self.densities[i - 1] + d) / 2 if i > 0 else d))] = i
        return x, array
