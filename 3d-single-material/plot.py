import matplotlib.pyplot as plt
import pandas as pd
import vtk
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


def plot_isosurface_3d(x, m_d, m_c, thresh):
    meshes = []
    for i, d in enumerate(m_d):
        array = np.zeros(x.shape)
        array[(x > ((m_d[i - 1] + d) / 2 if i > 0 else thresh)) &
              (x < (m_d[i + 1] + d) / 2 if i < len(m_d) - 1 else True)] = 1
        voxel = trimesh.voxel.VoxelGrid(array).as_boxes()
        meshes.append(Mesh([voxel.vertices, voxel.faces], c=m_c[i]))

    show_meshes = meshes.copy()

    def update_cut_plane(obj, event):
        print(f'normal={obj.GetNormal()}, origin={obj.GetOrigin()}')
        for j, m in enumerate(show_meshes):
            m.cut_with_plane(normal=obj.GetNormal(), origin=obj.GetOrigin())

    cutter = PlaneCutter(meshes[0],normal=(0, 1, 0), can_translate=False)
    s = cutter.clipper.GetInput()
    # cutter.widget.AddObserver("InteractionEvent", update_cut_plane)
    plot = Plotter(interactive=False)
    plot.show(meshes)
    plot.add(cutter)
    plot.interactive()


df = pd.read_excel('xs.xlsx', engine='openpyxl')
data = df.values
plot_isosurface_3d(np.reshape(data, (30, 30, 30), 'F'),
                   [0.6, 0.8, 1],
                   ['red', 'blue', 'green'], 0.5)
