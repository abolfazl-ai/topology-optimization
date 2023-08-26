import numpy as np
import pandas as pd
import trimesh
import vedo
from vedo import Volume, show, PlaneCutter, Mesh, Plotter


def plot_isosurface_3d(x, m_d, m_c, thresh):
    meshes = []
    for i, d in enumerate(m_d):
        array = np.zeros(x.shape)
        array[(x > ((m_d[i - 1] + d) / 2 if i > 0 else thresh)) &
              (x < (m_d[i + 1] + d) / 2 if i < len(m_d) - 1 else True)] = 1
        voxel = trimesh.voxel.VoxelGrid(array).as_boxes()
        meshes.append(Mesh([voxel.vertices, voxel.faces], c=m_c[i]))
    assem = vedo.Assembly(meshes)

    cutter = PlaneCutter(assem.actors[0], normal=(0, 1, 0), can_translate=False)
    plot = Plotter(interactive=False)
    plot.show(assem)
    plot.add(cutter)
    plot.interactive()


df = pd.read_excel('xs.xlsx', engine='openpyxl')
data = df.values
plot_isosurface_3d(np.reshape(data, (30, 30, 30), 'F'),
                   [0.6, 0.8, 1],
                   ['red', 'blue', 'green'], 0.5)
