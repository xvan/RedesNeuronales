import unittest
from typing import List

import numpy as np
import pandas as pd
import itertools
import pickle
import os

from multiprocessing import Pool

from tp2.multilayer import BackPropagationTrainer, MultilayerNetwork, BackPropagationMultistartTrainer
import tp2.aux as tp2Aux
from tp2.perceptron import TrainDataType


class SequentialTestMinibatch(unittest.TestCase):
    @unittest.skip("For Debugging Purposes")
    def test_one(self):
        training_samples, testing_samples = tp2Aux.Exercise4.generate_random_dataset(20, 0.8)
        ParallelTestMinibatch.train_minibatch(0, training_samples, testing_samples, 1, 0.05)


class ParallelTestMinibatch(unittest.TestCase):
    @staticmethod
    def minibatch_sizes_generator(n):
        yield n
        while n > 1:
            n = np.ceil(n/2)
            yield n

    def test_minibatch(self):
        training_samples, testing_samples = tp2Aux.Exercise4.generate_random_dataset(20, 0.8)
        #minibatch_sizes = list(self.minibatch_sizes_generator(len(training_samples)))
        minibatch_sizes = [6400, 1600, 400, 100, 25, 4, 1]
        #learning_rate = [0.01, 0.005, 0.001]
        learning_rate = [0.0005, 0.0001]

        param_combinations = [(n, training_samples, testing_samples, s, r)
                              for n, (s, r) in enumerate(itertools.product(minibatch_sizes, learning_rate))]
        print("total: ", len(param_combinations))
        with Pool(processes=8) as pool:  # run no more than 6 at a time
            TT = pool.starmap(self.train_minibatch, param_combinations)
            with open(os.path.expanduser("~/results.pkl"), "wb") as fo:
                pickle.dump(TT, fo)

    @staticmethod
    def train_minibatch(idx, training_samples, testing_samples, batch_size, learning_rate):
        print("starting:", idx, batch_size, learning_rate)
        np.seterr(all="ignore")
        layers = [3, 25, 1]
        mn = MultilayerNetwork(layers)
        mn.perceptrons[-1].activator = lambda x: x
        trainer = BackPropagationTrainer(mn, training_samples, batch_size)
        logger = EpochLogger(trainer, testing_samples)
        trainer.cost_callback = lambda x: logger.log_epoch(x)
        trainer.learning_rate = learning_rate
        trainer.iterations_limit = 20000
        trainer.cost_target = 0.01
        trainer.train()
        print("finished:", idx, trainer.best_cost, batch_size, learning_rate)
        return batch_size, learning_rate, trainer.best_weights, logger.result_costs

class EpochLogger:
    def __init__(self, multilayer_trainer: BackPropagationMultistartTrainer, test_samples: TrainDataType):
        self.multilayer_trainer = multilayer_trainer
        self.test_samples = test_samples
        self.training_costs: List[float] = []
        self.testing_costs: List[float] = []

    def log_epoch(self, cost: float):
        self.training_costs.append(cost)
        self.testing_costs.append(self.calculate_training_cost())

    def calculate_training_cost(self):
        return np.mean(self.multilayer_trainer.process_costs())

    @property
    def result_costs(self):
        return list(zip(self.training_costs, self.testing_costs))

if __name__ == '__main__':
    unittest.main()
