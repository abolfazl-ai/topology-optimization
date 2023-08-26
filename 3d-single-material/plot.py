import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from skimage.measure import marching_cubes


def plot_result(result, densities, colors, section_percentage=0):
    voxel_array = result > 0.6
    plot_colors = np.empty(voxel_array.shape, dtype=object)
    sec_colors = [(0, "white")]
    for i, d in enumerate(densities):
        plot_colors = colors[i]
        sec_colors.append((d / np.max(densities), colors[i]))

    section = result[:, min(int(np.floor(section_percentage * result.shape[2] / 100)), result.shape[2] - 1), :]
    sec_cmap = LinearSegmentedColormap.from_list("section", sec_colors)

    fig = plt.figure()
    plt.figaspect(1)
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.voxels(voxel_array, edgecolor='k', facecolors=plot_colors)
    ax1.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(section.T, cmap=sec_cmap, origin='lower', extent=[0, result.shape[0], 0, result.shape[2]])
    ax2.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    ax2.grid(True)

    plt.get_current_fig_manager().window.showMaximized()
    plt.show()


# xx = np.random.randint(0, 5, size=(30, 30, 30))
# xx[xx < 3] = 0
# dd = [1, 2, 3, 4]
# cc = ['red', 'blue', 'green', 'orange']
# plot_result(xx, dd, cc)

def plot_3d(array, iso_value=0.5):
    verts, faces, _, _ = marching_cubes(array, level=iso_value)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='viridis', edgecolor='none')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


from vedo import *
import trimesh

def plot_isosurface_3d(array, iso_value=0.5):
    vol = Volume(array).smooth_gaussian()
    lego = vol.legosurface(iso_value, 1)
    vertices, faces, _, _ = marching_cubes(array, level=iso_value)
    # trimesh_mesh = trimesh.Trimesh(vertices=lego.points(), faces=lego.faces())
    trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    solid = trimesh_mesh.voxelized(pitch=0.1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    occupied_voxels = np.array(np.where(solid))
    ax.scatter(occupied_voxels[0], occupied_voxels[1], occupied_voxels[2], c='b', marker='o', alpha=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

    # plot = Plotter(interactive=False)
    # plot.show(solid)
    # cutter = PlaneCutter(solid, normal=(0, 1, 0))
    # plot.add(cutter)
    # plot.interactive()


df = pd.read_excel('xs.xlsx', engine='openpyxl')
data = df.values
plot_isosurface_3d(np.reshape(data, (30, 30, 30), 'F'))
