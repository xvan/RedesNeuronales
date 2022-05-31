import unittest
import numpy as np
import pandas as pd
import itertools
import pickle

from multiprocessing import Pool

from tp2.multilayer import SingleAttemptMultilayerTrainer, MultilayerNetwork
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

        param_combinations = ((training_samples, s, r) for s, r in itertools.product(minibatch_sizes, learning_rate))

        with Pool(processes=8) as pool:  # run no more than 6 at a time
            TT = pool.starmap(self.train_minibatch, param_combinations)
            with open("~/results.pkl", "wb") as fo:
                pickle.dump(TT, fo)

    @staticmethod
    def train_minibatch(training_samples, batch_size, learning_rate):
        layers = [3, 25, 1]
        func_costs = []
        mn = MultilayerNetwork(layers)
        mn.perceptrons[-1].activator = lambda x: x
        trainer = SingleAttemptMultilayerTrainer(mn, training_samples, )
        trainer.cost_callback = lambda x: func_costs.append()
        trainer.learning_rate = learning_rate
        trainer.train()
        pd.DataFrame(func_costs, columns=["cost"]).to_csv("~/results_%s_%s.csv" % (str(batch_size), str(learning_rate)))
        return batch_size, learning_rate, trainer.best_weights(), func_costs

if __name__ == '__main__':
    unittest.main()
