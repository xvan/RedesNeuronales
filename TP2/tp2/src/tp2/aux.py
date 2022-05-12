import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tp2.perceptron import ThresholdUnit


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