import matplotlib.colors as mc
import numpy as np
import pyvista as pv
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


class Plotter:
    def __init__(self, path, materials, thresh=0.3, zoom=0, show_edges=False):
        self.x = np.load(path)
        self.dimension = len(self.x.shape)
        self.plotter = Plotter2D(self.x, materials) if self.dimension == 2 else Plotter3D(self.x, materials, thresh,
                                                                                          zoom, show_edges)

    def update(self, path, title=None, interactive=False):
        self.x = np.load(path)
        self.plotter.update(self.x, title, interactive)


class Plotter2D:
    def __init__(self, x, materials):
        plt.ion()
        self.densities, self.names, self.colors = materials['D'], materials['names'], materials['colors']
        self.densities[0] = 0
        self.fig, self.ax = plt.subplots()
        cmap = mc.LinearSegmentedColormap.from_list('mesh', list(zip(self.densities, self.colors)))
        custom_lines = [Line2D([0], [0], marker='o', label='Scatter',
                               lw=0, markerfacecolor=c, markersize=10) for c in self.colors]
        self.im = self.ax.imshow(x, origin='lower', cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
        self.ax.set_title(F'Iteration: {0}, Change: {1:0.4f}')
        self.ax.legend(custom_lines, self.names, ncol=len(self.colors))
        plt.get_current_fig_manager().window.showMaximized()
        plt.pause(0.1)
        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self.ax.draw_artist(self.im)
        self.fig.canvas.blit(self.fig.bbox)

    def update(self, x, title, interactive):
        array = x.copy()
        # for i, d in enumerate(self.densities):
        #     array[(x > ((self.densities[i - 1] + d) / 2 if i > 0 else d))] = d
        self.fig.canvas.restore_region(self.bg)
        self.im.set_array(array)
        if title is not None: self.ax.set_title(title)
        self.ax.draw_artist(self.im)
        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()
        if not interactive: plt.show(block=True)


class Plotter3D:

    def __init__(self, x, materials, thresh=0.3, zoom=0, show_edges=False):
        pv.global_theme.show_edges = show_edges
        self.densities, self.names, self.colors = materials['D'][1:], materials['names'][1:], materials['colors'][1:]
        self.x, self.interactive, self.init, self.threshold = np.moveaxis(x, -1, 0), False, False, thresh
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

    def update(self, x, title, interactive):
        if x is not None:
            self.x = np.moveaxis(x, -1, 0)
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
                  (self.x <= ((self.densities[i + 1] + d) / 2 if i < len(self.densities) - 1 else d))] = i + 1

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

    def export(self, file_path):
        self.roi.scale(1 / max(self.x.shape)).rotate_x(-90).extract_surface().save(file_path)

# def view(path, d, n, c, export=False):
#     p = Plot3D(path, d, n, c, show_edges=False)
#     p.update(path, interactive=False)
#     if export: p.export(path[0:-3] + 'stl')
#
#
# if __name__ == '__main__':
#     # density, name, colors = [0, 1], ['V', 'Solid'], ['w', 'g']
#     density, name, colors = [0, 1, 0.85], ['V', 'TPU', 'ABS'], ['w', 'r', 'b']
#     # density, name, colors = [0, 0.5, 1], ['V', 'PCL', 'PLA'], ['w', 'r', 'b']
#     view('runs/shear-distributed.npy', density, name, colors, )
