from abc import ABC, abstractmethod
import matplotlib.colors as mc
import numpy as np
import pyvista as pv
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import copy


def plotter(x, materials):
    return Plotter2D(x, copy.deepcopy(materials)) if len(x.shape) == 2 \
        else Plotter3D(x, copy.deepcopy(materials))


class Plotter(ABC):
    def __init__(self, x, materials):
        self.x = x if len(x.shape) == 2 else np.moveaxis(x, -1, 0)
        self.D, self.N, self.C = materials['D'], materials['names'], materials['colors']
        self.interactive, self.init, self.threshold, self.D[0] = False, False, 0.3, 0

    @abstractmethod
    def show(self, x=None, title='Optimization result', interactive=False):
        pass

    @abstractmethod
    def export(self, file_path):
        pass


class Plotter2D(Plotter):

    def __init__(self, x, materials):
        super().__init__(x, materials)
        plt.ion()
        self.fig, self.ax = plt.subplots()
        cmap = mc.LinearSegmentedColormap.from_list('mesh', list(zip(self.D, self.C)))
        custom_lines = [Line2D([0], [0], marker='o', label='Scatter',
                               lw=0, markerfacecolor=c, markersize=10) for c in self.C]
        self.im = self.ax.imshow(x, origin='lower', cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
        self.ax.legend(custom_lines, self.N, loc='upper center', ncol=len(self.C), bbox_to_anchor=(0.5, 1.1))
        # plt.get_current_fig_manager().window.showMaximized()
        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self.ax.draw_artist(self.im)
        self.fig.canvas.blit(self.fig.bbox)
        plt.draw()
        plt.pause(0.1)

    def show(self, x=None, title='Optimization result', interactive=False):
        if x is not None: self.x = x
        array = self.x.copy()
        self.fig.canvas.restore_region(self.bg)
        self.im.set_array(array)
        if title is not None: self.fig.canvas.manager.set_window_title(title)
        self.ax.draw_artist(self.im)
        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()
        if not interactive: plt.show(block=True)

    def export(self, file_path):
        pass


class Plotter3D(Plotter):

    def __init__(self, x, materials):
        super().__init__(x, materials)
        self.D.pop(0), self.C.pop(0), self.N.pop(0)
        self.grid = pv.ImageData(dimensions=np.array(self.x.shape) + 1)
        self.p = pv.Plotter(shape=(1, 2))
        self.p.add_mesh(self.grid, color='gray', opacity=0, name='blank')
        self.p.add_legend(list(zip(self.N, self.C)), 'w', loc='upper left', face=None, size=(0.15, 0.18))
        self.p.subplot(0, 1)
        self.p.add_slider_widget(self._th, [0, 1], self.threshold, 'Threshold', (0.1, 0.9), style='modern', fmt='%0.2f')
        self.p.add_bounding_box()
        self.p.link_views()

    def show(self, x=None, title='Optimization result', interactive=False):
        if x is not None: self.x = np.moveaxis(x, -1, 0)
        self.interactive = interactive
        self._show_data()
        if not self.init:
            self.p.camera_position = 'xz'
            self.p.camera.azimuth = 120
            self.p.camera.elevation = 15
            self.init = True
        self.p.show(title=title, full_screen=False, window_size=(1100, 550), interactive_update=interactive)

    def _th(self, value):
        if 'Density' in self.grid.point_data.keys():
            self.roi = self.grid.clip_scalar(value=value, scalars='Density', invert=False)
            if self.roi.points.size > 0:
                self.p.subplot(0, 1)
                self.p.add_mesh(self.roi, scalars='Color', cmap=self.C, name='isosurface', show_scalar_bar=False) \
                    if not self.interactive else self.p.add_volume(self.x, opacity=[0, 0.6], clim=[0.01, 1],
                                                                   cmap=self.C, name='isosurface',
                                                                   show_scalar_bar=False)

    def _show_data(self):
        array = self.x.copy()
        for i, d in enumerate(self.D):
            array[(self.x >= ((self.D[i] - 1 + d) / 2 if i > 0 else d)) &
                  (self.x <= ((self.D[i + 1] + d) / 2 if i < len(self.D) - 1 else d))] = i + 1

        self.grid.cell_data['Density'] = self.x.flatten(order="F")
        self.grid.cell_data['Color'] = array.flatten(order="F")
        self.grid = self.grid.cell_data_to_point_data()
        self.grid = self.grid.gaussian_smooth(scalars='Color')
        self.roi = self.grid.clip_scalar(value=self.threshold, scalars='Density', invert=False)

        if self.roi.points.size > 0:
            self.p.subplot(0, 0)
            if not self.interactive:
                self.p.add_mesh(self.roi, style='wireframe', color='gray', opacity=0.05, name='wireframe')
                self.p.add_mesh_clip_plane(self.roi, 'x', True, origin=np.array(self.x.shape) * 2, assign_to_axis='x',
                                           scalars='Color', cmap=self.C, name='clip', show_scalar_bar=False)
            else:
                self.p.add_mesh(self.roi, scalars='Color', cmap=self.C, name='wireframe', show_scalar_bar=False)
            self.p.add_axes()
            self._th(self.threshold)

    def export(self, file_path):
        self.roi.scale(1 / max(self.x.shape)).rotate_x(-90).extract_surface().save(file_path)


def view(path, materials):
    x = np.load(path)
    p = plotter(x, materials)
    p.show()


if __name__ == '__main__':
    # mat = {'names': ['V', 'Solid'], 'colors': ['w', 'b'], 'D': [0, 1]}
    mat = {'names': ['V', 'TPU', 'ABS'], 'colors': ['w', 'r', 'g'], 'D': [0, 1, 0.85]}
    # mat = {'names': ['V', 'PCL', 'PLA'], 'colors': ['w', 'r', 'b'], 'D': [0, 0.5, 1]}
    # mat = {'names': ['V', 'Solid'], 'colors': ['w', 'k'], 'D': [0, 1]}
    view('runs/shear-concentrated.npy', mat, )
