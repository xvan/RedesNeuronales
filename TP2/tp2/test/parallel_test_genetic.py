import unittest

from test.test_tp2 import xor_gate_table
from tp2.capacity_estimator import IncrementalEstimator
from tp2.multilayer import GeneticTrainer, MultilayerNetwork


class Counter:
    count = 0

    def increment(self):
        self.count += 1


class ParallelGeneticTest(unittest.TestCase):
    def test_genetics(self):
        ie = self.get_estimation()
        print(ie.mean)

    def train_genetic(self):
        layers = [2, 2, 1]
        mn = MultilayerNetwork(layers)
        trainer = GeneticTrainer(mn, xor_gate_table)
        counter = Counter()
        # mn.perceptrons[-1].activator = lambda x: (x >= 0).astype(int)
        trainer.fitness_callback = lambda _: counter.increment()
        trainer.train()
        return counter.count, trainer.target_reached

    def get_estimation(self) -> IncrementalEstimator:
        ie = IncrementalEstimator()
        while True:
            cnt, finished = self.train_genetic()
            print(cnt)
            if finished:
                ie.append(cnt)
            if ie.disc(1.96) < 0.01:
                break
        return ie

if __name__ == '__main__':
    unittest.main()
