import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from tp3.kohonen import KohonenNetwork, CircularDataGenerator, DataGenerator, SquareDataGenerator
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


def plot_map_for_tsp(n: int):
    generator=SquareDataGenerator()
    target = generator.generate(n)

    plt.scatter(target[:, 0], target[:, 1], c="orange")
    _ = plt.title("Distribución %s uniforme" % generator.title)

    plt.figure()
    kn = TspAutomapper(n)
    kn.set_target(target)
    plot_map(kn)
    _ = plt.title("Pesos semillas, en mapa de 1x%i" % n)

    kn.train(1000)

    plt.figure()
    plot_map(kn)
    plt.scatter(target[:, 0], target[:, 1], c="orange")
    _ = plt.title("Mapa auto-organizado")
