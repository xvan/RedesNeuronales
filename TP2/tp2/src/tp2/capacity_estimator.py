import itertools
import numpy as np
from typing import List
import pickle

from tp2.perceptron import Perceptron, TooManyIterations, TrainDataType


class CapacityEstimator:
    def __init__(self, perceptron: Perceptron):
        self.perceptron = perceptron

    def brute_capacity(self, dimension, patterns):
        space = list(itertools.product([1, -1], repeat=dimension))

        learned = 0
        total = 0
        for X in itertools.combinations(space, patterns):
            for Y in itertools.product([1, -1], repeat=len(X)):
                total += 1
                data = [(x, (y,)) for x, y in zip(X, Y)]
                try:
                    self.perceptron.train(data)
                    learned += 1
                except TooManyIterations:
                    pass

        return learned/total

    def generate_data(self, dimension: int, patterns: int) -> TrainDataType:
        rand_values = np.random.rand(patterns, dimension + 1) * 2 - 1
        return [(row[:-1], np.sign(row[-1:])) for row in rand_values]

    def capacity(self, dimension: int, patterns: int, iterations_limit: int) -> float:
        ie = IncrementalEstimator()

        while True:
            for batch in range(100):
                try:
                    data = self.generate_data(dimension, patterns)
                    self.perceptron.train(data, iterations_limit=iterations_limit)
                    ie.append(1)
                except TooManyIterations:
                    ie.append(0)

            if ie.disc(1.96) < 0.01:
                break

        print("d:%i, p:%i, n:%i, u: %.03f, disc: %0.3f" % (dimension, patterns, ie.count, ie.mean, ie.disc(1.96)))
        return ie.mean


class IncrementalEstimator:

    def __init__(self):
        self._Sn: float = 0
        self._mean: float = 0
        self.count: int = 0

    @property
    def mean(self) -> float:
        return self._mean if self.count > 0 else np.nan

    @property
    def variance(self) -> float:
        return self._Sn / (self.count - 1) if self.count >= 2 else np.nan

    @property
    def std(self) -> float:
        return np.sqrt(self.variance)

    def disc(self, confidence_factor: float):
        return confidence_factor * self.std / np.sqrt(self.count)

    def append_range(self, data: List[float]):
        for x in data:
            self.append(x)

    def append(self, x: float):
        self.count += 1
        old_mean = self._mean
        self._mean += (x - self.mean)/self.count
        self._Sn += (x - self.mean)*(x-old_mean)

