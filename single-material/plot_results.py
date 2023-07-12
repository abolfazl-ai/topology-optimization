import copy
import matplotlib.pyplot as plt


def plot_output(nodes, elements, d, exaggeration=0):
    if exaggeration == 0:
        exaggeration = 0.1 / max(abs(d))

    fig, ax = plt.subplots()

    new_nodes = copy.deepcopy(nodes)
    for n, node in new_nodes.items():
        new_position = (node.position[0] + exaggeration * d[int(2 * (n - 1))],
                        node.position[1] + exaggeration * d[int(2 * (n - 1) + 1)])
        node.position = new_position

    for i in range(elements.shape[1]):
        for j in range(elements.shape[0]):
            new_node_positions = [new_nodes[node].position for node in elements[i, j].nodes]
            old_node_positions = [nodes[node].position for node in elements[i, j].nodes]
            ax.plot(*zip(*old_node_positions, old_node_positions[0]), color='blue', zorder=1)
            ax.plot(*zip(*new_node_positions, new_node_positions[0]), color='black', zorder=1)

    ax.scatter(*zip(*[node.position for node in nodes.values()]), marker='o', linewidths=0.5, color='blue', zorder=2)
    ax.scatter(*zip(*[node.position for node in new_nodes.values()]), marker='o', linewidths=0.5, color='red', zorder=2)

    for node in nodes.values():
        ax.annotate(node.number, node.position, textcoords="offset points", xytext=(-6, -8), ha='center', fontsize=6)

    ax.set(xlabel="X", ylabel="Y", title='Elements after deformation', aspect=1,
           xlim=(ax.get_xlim()[0] - 1, ax.get_xlim()[1] + 1), ylim=(ax.get_ylim()[0] - 0.5, ax.get_ylim()[1] + 0.5))

    plt.show()
