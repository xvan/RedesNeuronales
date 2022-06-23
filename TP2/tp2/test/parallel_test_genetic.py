import os
import pickle
import unittest

from test.test_tp2 import xor_gate_table
from tp2.capacity_estimator import IncrementalEstimator
from tp2.multilayer import GeneticTrainer, MultilayerNetwork
from multiprocessing import Pool
import itertools


class Counter:
    count = 0

    def increment(self):
        self.count += 1


class ParallelGeneticTest(unittest.TestCase):

    @unittest.skip("For Debugging Purposes")
    def test_genetics(self):
        # ie, se = self.get_estimation(10, 0.01, 0.2)
        ie, se = self.get_estimation(0, 10, 0, 0.01)
        print(ie.mean, se.count)

    def test_genetics_parallel(self):
        generation_specimens = [10, 20, 100]
        cross_probability = [0, 0.25, 0.5]
        mutation_std = [0.05, 0.1, 0.5]

        param_combinations = [(n, g, c, m) for n, (g, c, m) in
                              enumerate(itertools.product(generation_specimens, cross_probability, mutation_std))]
        print(param_combinations)
        print("total: ", len(param_combinations))
        with Pool(processes=8) as pool:  # run no more than 6 at a time
            TT = pool.starmap(self.get_estimation, param_combinations)
            with open(os.path.expanduser("~/results.pkl"), "wb") as fo:
                pickle.dump(TT, fo)

    @staticmethod
    def get_estimation(test_index: int, pool_size: int, cross_over_probability: float,
                       mutation_std: float) -> [IncrementalEstimator, IncrementalEstimator]:

        iterations_estimator = IncrementalEstimator()
        success_estimator = IncrementalEstimator()
        with open(os.path.expanduser("~/iteration_%i.csv" % test_index), "w") as fo:
            while True:
                cnt, finished = ParallelGeneticTest.train_genetic(pool_size, cross_over_probability, mutation_std)
                success_estimator.append(finished)
                if finished:
                    iterations_estimator.append(cnt)
                if ParallelGeneticTest.one_percent_error(iterations_estimator):
                    break
                if ParallelGeneticTest.abort_condition(success_estimator):
                    break

                loop_log = "%i:, cnt:%i, ratio:%.2f, dsc:%.2f > mean:%.2f / 100" % (
                    test_index, success_estimator.count, success_estimator.mean,
                    iterations_estimator.disc(1.96), iterations_estimator.mean, )
                print(loop_log)
                fo.write(loop_log + "\n")

        return {"test_index": test_index, "pool_size": pool_size, "cross_over": cross_over_probability,
                "mutation": mutation_std, "iterations_estimator": iterations_estimator,
                "success_estimator": success_estimator}

    @staticmethod
    def train_genetic(pool_size, cross_over_probability, mutation_std):
        layers = [2, 5, 1]
        mn = MultilayerNetwork(layers)
        trainer = GeneticTrainer(mn, xor_gate_table)
        counter = Counter()
        trainer.fitness_callback = lambda _: counter.increment()
        trainer.max_iterations = 100000

        trainer.pool_size = pool_size
        trainer.mutation_std = mutation_std
        trainer.cross_over_probability = cross_over_probability

        trainer.train()
        return counter.count, trainer.target_reached

    @staticmethod
    def abort_condition(ie: IncrementalEstimator) -> bool:
        if ie.count < 100:
            return False
        if ie.disc(1.96) > 0.01:
            return False
        return ie.mean < 0.8

    @staticmethod
    def one_percent_error(ie: IncrementalEstimator) -> bool:
        return ie.disc(1.96) < ie.mean / 100


if __name__ == '__main__':
    unittest.main()
