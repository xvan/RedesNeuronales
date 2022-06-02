import unittest
from typing import List

import copy
import numpy as np
import pandas as pd
import itertools
import pickle
import os

from multiprocessing import Pool, get_context, current_process

from tp2.perceptron import TrainDataType
from tp2.multilayer import SingleAttemptMultilayerTrainer, MultilayerNetwork, MultilayerTrainer
import tp2.aux as tp2Aux


class ParallelTestMinibatch(unittest.TestCase):

    @staticmethod
    def minibatch_sizes_generator(n):
        yield n
        while n > 1:
            n = np.ceil(n/2)
            yield n

    def test_minibatch(self):
        training_samples, testing_samples = tp2Aux.Exercise4.generate_dataset(20, 0.8)
        minibatch_sizes = list(self.minibatch_sizes_generator(len(training_samples)))
        learning_rate = [0.1, 0.05, 0.01, 0.005, 0.001]

        param_combinations = ((copy.deepcopy(training_samples), copy.deepcopy(testing_samples), s, r) for s, r in itertools.product(minibatch_sizes, learning_rate))
        with get_context('fork').Pool(processes=4) as pool:  # run no more than 6 at a time
            TT = pool.map(ParallelTestMinibatch.train_minibatch_starmap, param_combinations)
            with open(os.path.expanduser("~/results.pkl"), "wb") as fo:
                pickle.dump(TT, fo)

    @staticmethod
    def train_minibatch_starmap(task_data):
        print(current_process().name, task_data[2:])
        return ParallelTestMinibatch.train_minibatch(*task_data)

    @staticmethod
    def train_minibatch(training_samples, testing_samples, batch_size, learning_rate):
        print("starting", batch_size, learning_rate)
        np.seterr(all="ignore")
        layers = [3, 25, 1]
        mn = MultilayerNetwork(layers)
        mn.perceptrons[-1].activator = lambda x: x
        trainer = SingleAttemptMultilayerTrainer(mn, training_samples, batch_size)
        epoch_logger = EpochLogger(trainer, testing_samples)
        trainer.epoch_callback = lambda x: epoch_logger.log_epoch(x)
        trainer.learning_rate = learning_rate
        trainer.iterations_limit = 20000
        trainer.train()
        costs_list = list(zip(epoch_logger.training_costs, epoch_logger.testing_costs))
        print("finished", batch_size, learning_rate)
        return batch_size, learning_rate, trainer.best_weights,  costs_list

    #@unittest.skip("For Debugging Purposes")
    def test_one(self):
        training_samples, testing_samples = tp2Aux.Exercise4.generate_dataset(20, 0.8)
        self.train_minibatch(training_samples, testing_samples, 3200, 0.1)

class EpochLogger:
    def __init__(self, multilayer_trainer: MultilayerTrainer, test_samples: TrainDataType):
        self.multilayer_trainer = multilayer_trainer
        self.test_samples = test_samples
        self.training_costs: List[float] = []
        self.testing_costs: List[float] = []

    def log_epoch(self, cost: float):
        self.training_costs.append(cost)
        self.testing_costs.append(self.calculate_training_cost())

    def calculate_training_cost(self):
        return np.mean([self.multilayer_trainer._set_network_states(x, y) for x, y in self.test_samples])


if __name__ == '__main__':
    unittest.main()
