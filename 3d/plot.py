import numpy as np
import pyvista as pv


class Plot3D:
    def __init__(self, path, density, name, colors, thresh=0.3, zoom=0, show_edges=False):
        pv.global_theme.show_edges = show_edges
        self.path, self.interactive, self.init = path, False, False
        self.threshold, self.densities, self.names, self.colors = thresh, density[1:], name[1:], colors[1:]
        self.x = np.load(path)
        self.grid = pv.ImageData(dimensions=np.array(self.x.shape) + 1)
        self.volume = self.grid.volume
        self.p = pv.Plotter(shape=(1, 2))
        self.p.add_mesh(self.grid, color='gray', opacity=0, name='blank')
        if zoom != 0: self.p.camera.zoom(zoom)
        self.p.subplot(0, 1)
        self.p.add_slider_widget(self._th, [0, 1], thresh, 'Threshold', (0.1, 0.9), style='modern', fmt='%0.2f')
        self.p.add_bounding_box()
        self.p.link_views()
        self.init = True

    def update(self, path=None, interactive=False):
        if path is not None:
            self.x = np.load(path)
        self.interactive = interactive
        self._show_data()
        self.p.show(full_screen=False, window_size=(1100, 550), interactive_update=interactive)

    def _th(self, value):
        if self.init:
            self.roi = self.grid.clip_scalar(value=value, scalars='Density', invert=False)
            if self.roi.points.size > 0:
                self.p.subplot(0, 1)
                self.p.add_text(f'Volume = {100 * self.roi.volume / self.volume:0.1f}%',
                                position=(30, 40), font_size=10, name='volume')
                if not self.interactive:
                    self.p.add_mesh(self.roi, scalars='Color', cmap=self.colors,
                                    show_scalar_bar=False, name='isosurface')
                else:
                    self.p.add_volume(self.x, opacity=[0, 0.6], clim=[0.01, 1], cmap=self.colors,
                                      show_scalar_bar=False, name='isosurface')

    def _show_data(self):
        array = self.x.copy()
        for i, d in enumerate(self.densities):
            array[(self.x >= ((self.densities[i - 1] + d) / 2 if i > 0 else d)) &
                  (self.x < ((self.densities[i + 1] + d) / 2 if i < len(self.densities) - 1 else d))] = i + 1

        self.grid.cell_data['Density'] = self.x.flatten(order="F")
        self.grid.cell_data['Color'] = array.flatten(order="F")
        self.grid = self.grid.cell_data_to_point_data()
        # self.grid = self.grid.gaussian_smooth(scalars='Color')
        self.roi = self.grid.clip_scalar(value=self.threshold, scalars='Density', invert=False)

        if self.roi.points.size > 0:
            self.p.subplot(0, 0)
            self.p.add_legend(list(zip(self.names, self.colors)), 'w', loc='upper left', face=None, size=(0.15, 0.18))
            if not self.interactive:
                self.p.add_mesh(self.roi, style='wireframe', color='gray', opacity=0.05, name='wireframe')
                self.p.add_mesh_clip_plane(self.roi, 'x', True, origin=np.array(self.x.shape) * 2, assign_to_axis='x',
                                           scalars='Color', cmap=self.colors, show_scalar_bar=False, name='clip')
            else:
                self.p.add_mesh(self.roi, scalars='Color', cmap=self.colors, show_scalar_bar=False, name='wireframe')
            self.p.add_axes()
            self._th(self.threshold)


def view(path):
    p = Plot3D(path, [0, 1, 0.85], ['V', 'TPU', 'ABS'], ['w', 'r', 'b'], show_edges=True)
    p.update(path, interactive=False)


# view('final.npy')
