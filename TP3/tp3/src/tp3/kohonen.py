from typing import List

import numpy as np


class CircularDataGenerator:
    def generate(self, n):
        uniform_2d = np.random.rand(n, 2)
        complex_circle = uniform_2d[:, 1] * np.exp(2j * np.pi * uniform_2d[:, 0])
        return np.array([complex_circle.real, complex_circle.imag]).transpose()


class KohonenNetwork:
    def __init__(self, shape):
        self.shape: List[int] = shape
        self.source_map: np.ndarray = np.dstack(np.meshgrid(*[np.arange(n) for n in self.shape]))
        self.weights_map: np.ndarray = None
        self.target: np.ndarray = None

    def set_target(self, target: List[List[float]]):
        self.target = target
        self.initialize_weights_map()

    def train(self,n):
        sigma = 0.5
        nabla = 1
        for _ in range(n):
            for sample_index in np.random.permutation(len(self.target)):
                pattern = self.target[sample_index]
                index = self.minimum_distance(pattern)
                distances = np.linalg.norm(self.source_map - self.source_map[index], axis=-1)
                vecindad = np.exp(-(distances ** 2)/2*sigma)
                delta_weight = -1 * nabla * (vecindad.transpose() * (self.weights_map - pattern).transpose()).transpose()
                self.weights_map += delta_weight
                sigma *= 0.999

    def minimum_distance(self, pattern):
        distances = self.distances(pattern)
        return np.unravel_index(np.argmin(distances), distances.shape)

    def distances(self, pattern):
        distance = np.linalg.norm(self.weights_map - pattern, axis=-1)
        return distance

    def initialize_weights_map(self):
        target_max, target_min = np.max(self.target, axis=0), np.min(self.target, axis=0)
        self.weights_map = (target_max - target_min) * np.random.rand(*self.weights_map_shape) + target_min

    @property
    def weights_map_shape(self):
        return list(self.shape) + [self.target_dim]

    @property
    def target_dim(self):
        return np.shape(self.target)[1]
