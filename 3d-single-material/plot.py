import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


def plot_result(result, densities, colors, section_percentage=0):
    voxel_array = result > 0
    plot_colors = np.empty(voxel_array.shape, dtype=object)
    sec_colors = [(0, "white")]
    for i, d in enumerate(densities):
        plot_colors[result == d] = colors[i]
        sec_colors.append((d / np.max(densities), colors[i]))

    section = result[:, min(int(np.floor(section_percentage * result.shape[2] / 100)), result.shape[2] - 1), :]
    sec_cmap = LinearSegmentedColormap.from_list("section", sec_colors)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.voxels(voxel_array, edgecolor='k', facecolors=plot_colors)
    ax1.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(section.T, cmap=sec_cmap, origin='lower', extent=[0, result.shape[0], 0, result.shape[2]])
    ax2.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    ax2.grid(True)

    plt.show()


xx = np.random.randint(0, 4, size=(10, 10, 10))
dd = [1, 2, 3]
cc = ['red', 'blue', 'green']
plot_result(xx, dd, cc)
