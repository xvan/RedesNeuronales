import pickle

import numpy as np
import itertools
import copy
import pandas as pd
import matplotlib.pyplot as plt
from tp2.perceptron import ThresholdUnit, TrainDataType
from tp2.multilayer import BackPropagationMultistartTrainer


def train_data_to_df(data):
    df = pd.DataFrame(data, columns=["entrada", "y"]).set_index("entrada")
    df.index = pd.MultiIndex.from_tuples(df.index)
    df.index.names = ["x%i" % (x + 1) for x in range(len(df.index.levels))]
    return df.explode("y")


def weights_to_df(tu: ThresholdUnit):
    return pd.DataFrame(tu.weights, columns=["w"]).rename_axis("x")


def plot_2d_tu(tu: ThresholdUnit):
    df = train_data_to_df(tu.train_data)
    plain_df = df.reset_index()
    plt.scatter(x=plain_df["x1"], y=plain_df["x2"], c=plain_df["y"])

    w = tu.weights
    recta = lambda x: -(w[0] * -1 + w[1] * x) / w[2]

    plt.plot([2, -2], [recta(2), recta(-2)])
    plt.xlim([-1.2, 1.2])
    plt.ylim([-1.2, 1.2])
    plt.xlabel("x1")
    plt.ylabel("x2")

def plot_row_generator(columns):
    while True:
        f, subplots = plt.subplots(1, columns)
        for subplot in subplots.ravel():
            yield subplot


def plot_all_cuts(trainer: BackPropagationMultistartTrainer, data: TrainDataType):
    W = copy.deepcopy(trainer.best_weights)
    weigth_coords = [(x, y) for x in range(len(W)) for y in np.ndindex(W[x].shape)]

    max_weight = np.ceil(np.max(np.abs([y for x in W for y in x.reshape(-1)])))

    mesh_size = 101
    space = np.linspace(-max_weight, max_weight, mesh_size)
    coord_combination = list(itertools.combinations(weigth_coords, 2))

    for (weight_coord_x, weight_coord_y), subplot in zip(coord_combination, plot_row_generator(3)):
        zz = np.zeros((mesh_size, mesh_size))
        for xi, x in enumerate(space):
            for yi, y in enumerate(space):
                trainer.network.perceptrons[weight_coord_x[0]].weights[weight_coord_y[1]] = x
                trainer.network.perceptrons[weight_coord_y[0]].weights[weight_coord_y[1]] = y
                zz[(xi, yi)] = np.sum([trainer._set_network_states(x, y) for x, y in data])

        im = subplot.contourf(space, space, zz, cmap="PuRd", vmin=0, vmax=12)
        #im = subplot.imshow(zz, cmap='PuRd', extent=[-max_weight, max_weight, -max_weight, max_weight], interpolation='bilinear', origin='lower')
        subplot.set_xlabel("weight coord:" + str(weight_coord_x))
        subplot.set_ylabel("weight coord:" + str(weight_coord_y))
        subplot.set_aspect('equal')
        plt.colorbar(im, ax=subplot, fraction=0.046, pad=0.04)
        trainer.best_weights = copy.deepcopy(W)
        trainer._restore_best_weights()


def plot_all_cuts_per_sample(trainer: BackPropagationMultistartTrainer, data: TrainDataType):
    W = copy.deepcopy(trainer.best_weights)
    weigth_coords = [(x, y) for x in range(len(W)) for y in np.ndindex(W[x].shape)]

    max_weight = np.ceil(np.max(np.abs([y for x in W for y in x.reshape(-1)])))

    mesh_size = 101
    space = np.linspace(-max_weight, max_weight, mesh_size)
    coord_combination = list(itertools.combinations(weigth_coords, 2))

    for (weight_coord_x, weight_coord_y) in coord_combination:
        figure, plotRows = plt.subplots(1, len(data))
        for (v, h), subplot in zip(data, plotRows.ravel()):
            zz = np.zeros((mesh_size, mesh_size))
            for xi, x in enumerate(space):
                for yi, y in enumerate(space):
                    trainer.network.perceptrons[weight_coord_x[0]].weights[weight_coord_y[1]] = x
                    trainer.network.perceptrons[weight_coord_y[0]].weights[weight_coord_y[1]] = y
                    zz[(xi, yi)] = trainer._set_network_states(v, h)

            im = subplot.contourf(space, space, zz, cmap="PuRd", vmin=0, vmax=5)
            #im = subplot.imshow(zz, cmap='PuRd', extent=[-max_weight, max_weight, -max_weight, max_weight], interpolation='bilinear', origin='lower')
            subplot.set_xlabel("weight coord:" + str(weight_coord_x))
            subplot.set_ylabel("weight coord:" + str(weight_coord_y))
            subplot.set_title("input:" + str(v))
            #subplot.set_aspect('equal')

        plt.colorbar(im, ax=plotRows.ravel().tolist(),fraction=0.046, pad=0.04)
        # divider = make_axes_locatable(ax1)
        # cax1 = divider.append_axes("right", size="5%", pad=0.05)
        # cbar = plt.colorbar(PC, cax=cax1)
        trainer.best_weights = copy.deepcopy(W)
        trainer._restore_best_weights()


class Exercise4:
    @staticmethod
    def target_function(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        return (np.sin(x) + np.cos(y) + z)/3.0

    @staticmethod
    def generate_samples(samples_per_dimension: int) -> np.ndarray:
        xv, yv, zv = Exercise4.generate_mesh(samples_per_dimension)
        kk = Exercise4.target_function(xv, yv, zv)
        return np.array([xv.ravel(), yv.ravel(), zv.ravel(), kk.ravel()]).transpose()

    @staticmethod
    def split_training_set(samples: np.ndarray, split_ratio: float) -> (np.ndarray, np.ndarray):
        np.random.shuffle(samples)
        return np.split(samples, [int(np.ceil(len(samples) * split_ratio))])

    @staticmethod
    def generate_mesh(samples_per_dimension):
        x = y = np.linspace(0, 2*np.pi, samples_per_dimension, endpoint=True)
        z = np.linspace(0, 1, samples_per_dimension, endpoint=True)
        xv, yv, zv = np.meshgrid(x, y, z)
        return xv, yv, zv

    @staticmethod
    def generate_regular_dataset(samples_per_dim: int, split_ratio: float) -> (np.ndarray, np.ndarray):
        samples = Exercise4.generate_samples(samples_per_dim)
        return tuple(Exercise4.numpy_to_training_data(subset)
                     for subset in Exercise4.split_training_set(samples, split_ratio))

    @staticmethod
    def generate_random_dataset(samples_per_dim: int, split_ratio: float) -> (np.ndarray, np.ndarray):
        samples = np.random.rand(samples_per_dim**3, 3)
        samples[:, :2] *= 2*np.pi

        samples = np.hstack([samples, Exercise4.target_function(samples[:, 0], samples[:, 1], samples[:, 2]).reshape(-1, 1)])

        return tuple(Exercise4.numpy_to_training_data(subset)
                     for subset in Exercise4.split_training_set(samples, split_ratio))

    @staticmethod
    def numpy_to_training_data(samples: np.ndarray) -> TrainDataType:
        return list(zip(samples[:, :-1], samples[:, -1:]))

    @staticmethod
    def load_pickled_data():
        #tuple (Hidden Layer Size, Pkl Path)
        sim_data_files = [(100, "data/minibatch_100h_20000e.pkl"),
                          (25, "data/minibatch_25h_20000e_LL.pkl"),
                          (15, "data/minibatch_15h_20000e.pkl"),
                          (25, "data/minibatch_25h_1000e.pkl")]

        sim_data = []
        for nn, fname in sim_data_files:
            with open(fname, "rb") as fi:
                sim_data += [sim_run[:3] + (np.array(sim_run[3]) / 3, nn) for sim_run in pickle.load(fi)]

        return sim_data

    @staticmethod
    def plot_sim_data(sim_data, legend_col=1):
        for data in sim_data:
            error = [pair[0] for pair in data[3]]
            plt.plot(error, label="(%i, %i, %.4f)" % (data[4], data[0], data[1]))
        plt.yscale("log")
        plt.xscale("log")
        plt.xlabel("epoch")
        plt.ylabel("cost")
        plt.legend(title="hidden layer, minibatch, learning rate", ncol=legend_col)
        plt.title("sin(x)+cos(y)+z")