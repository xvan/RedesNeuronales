import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from tp3.kohonen import KohonenNetwork, CircularDataGenerator, DataGenerator, SquareDataGenerator, KohonenClustering
from tp3.tsp import TspAutomapper


def plot_map(kn: KohonenNetwork):
    xx = kn.weights_map[:, :, 0]
    yy = kn.weights_map[:, :, 1]

    plt.scatter(xx, yy, alpha=0.5)

    segs1 = np.stack((xx, yy), axis=2)
    segs2 = segs1.transpose(1, 0, 2)
    plt.gca().add_collection(LineCollection(segs1))
    plt.gca().add_collection(LineCollection(segs2))


def plot_map_for_distribution(generator: DataGenerator, shape=[5, 5]):
    target = generator.generate(1000)

    plt.scatter(target[:, 0], target[:, 1], c="orange")
    _ = plt.title("Distribución %s uniforme" % generator.title)

    plt.figure()
    kn = KohonenNetwork(shape)
    kn.set_target(target)
    plot_map(kn)
    _ = plt.title("Pesos semillas, en mapa de 5x5")

    kn.train(400)

    plt.figure()
    plt.scatter(target[:, 0], target[:, 1], c="orange")
    plot_map(kn)
    _ = plt.title("Mapa auto-organizado")


def plot_map_for_tsp(cities: int):

    grids = (np.array([1.0, 1.5, 2.0, 4.0]) * cities).astype(int)

    generator = SquareDataGenerator()
    target = generator.generate(cities)

    plt.scatter(target[:, 0], target[:, 1], c="orange")
    _ = plt.title("Distribución %s uniforme" % generator.title)

    for grid in grids:
        plt.figure()
        kn = TspAutomapper(grid)
        kn.set_target(target)
        plot_map(kn)
        _ = plt.title("Pesos semillas, en mapa de 1x%i" % grid)

        kn.train(1000)

        plt.figure()
        plot_map(kn)
        plt.scatter(target[:, 0], target[:, 1], c="orange")
        _ = plt.title("Mapa auto-organizado de 1x%i" % grid)


def plot_hitmap(hitmap: np.ndarray, with_tags = True):
    fig, ax = plt.subplots()
    im = ax.imshow(hitmap)

    # Loop over data dimensions and create text annotations.
    for idx in np.ndindex(hitmap.shape):
        text = ax.text(idx[1], idx[0], ("%.2f" % hitmap[idx]).replace(".00",""),
                       ha="center", va="center")


def plot_mesh(kc: KohonenClustering):
    fig, ax = plt.subplots()
    dx,dy = kc.kn.shape
    x = np.arange(dx)
    y = np.arange(dy)
    xx, yy = np.meshgrid(x, y)

    for x in range(dx):
        lines = np.c_[xx[x, :-1], yy[x, :-1], xx[x, 1:], yy[x, 1:]]
        z = np.linalg.norm(kc.kn.weights_map[x, :-1, :] - kc.kn.weights_map[x, 1:, :], axis=-1)
        lc = LineCollection(lines.reshape(-1, 2, 2), array=z, linewidths=2)
        ax.add_collection(lc)

    for y in range(dy):
        lines = np.c_[xx[:-1,y], yy[:-1,y], xx[1:,y], yy[1:,y]]
        z = np.linalg.norm(kc.kn.weights_map[:-1, y, :] - kc.kn.weights_map[1:, y, :], axis=-1)
        lc = LineCollection(lines.reshape(-1, 2, 2), array=z, linewidths=2, cmap="hot")
        ax.add_collection(lc)

    ax.scatter(xx, yy, c="orange")