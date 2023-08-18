import matplotlib.pyplot as plt
import numpy as np


def plot_result(result, densities, colors, section_percentage=50):
    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(5.5, 3.5), layout="constrained")

    plt.style.use('_mpl-gallery')

    voxel_array = result > 0
    plot_colors = np.empty(voxel_array.shape, dtype=object)
    for i, d in enumerate(densities):
        plot_colors[result == d] = colors[i]

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.voxels(voxel_array, edgecolor='k', facecolors=plot_colors)
    plt.show()


xx = np.random.randint(0, 4, size=(28, 28, 28))
dd = [1, 2, 3]
cc = ['red', 'blue', 'green']
plot_result(xx, dd, cc)
