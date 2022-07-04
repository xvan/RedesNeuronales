from abc import ABCMeta, abstractmethod
from typing import List

import numpy as np


class DataGenerator(metaclass=ABCMeta):
    @abstractmethod
    def generate(self, n):
        pass

    @property
    @abstractmethod
    def title(self):
        pass


class CircularDataGenerator(DataGenerator):
    @property
    def title(self):
        return "Circular"

    def generate(self, n):
        uniform_2d = np.random.rand(n, 2)
        complex_circle = np.sqrt(uniform_2d[:, 1]) * np.exp(2j * np.pi * uniform_2d[:, 0])
        return np.array([complex_circle.real, complex_circle.imag]).transpose()


class SquareDataGenerator(DataGenerator):
    @property
    def title(self):
        return "Cuadrada"

    def generate(self, n):
        return 2 * np.random.rand(n, 2) - 1


class TriangleDataGenerator(DataGenerator):
    @property
    def title(self):
        return "Triangular"

    def generate(self, n):
        uniform_2d = np.random.rand(n, 2)
        uniform_2d[:, 0] = np.sqrt(uniform_2d[:, 0])
        uniform_2d[:, 1] *= uniform_2d[:, 0]
        return uniform_2d


class KohonenNetwork:
    def __init__(self, shape):
        self.shape: List[int] = shape
        self.source_map: np.ndarray = np.squeeze(np.dstack(np.meshgrid(*[np.arange(n) for n in self.shape])))
        self.weights_map: np.ndarray = None
        self.target: np.ndarray = None
        self.final_sigma = 0.1
        self.start_sigma = 1
        self.sigma = self.start_sigma
        self.nabla = self.sigma
        self.step_coeff = None

    def set_target(self, target: List[List[float]]):
        self.target = target
        self.initialize_weights_map()

    def train(self, n):
        self.init_coeffs(n)
        for _ in range(n):
            for sample_index in np.random.permutation(len(self.target)):
                self.process_pattern(self.target[sample_index])

    def process_pattern(self, pattern):
        index = self.minimum_distance(pattern)
        self.weights_map += self.delta_weights(index, pattern)
        self.update_coeff()

    def delta_weights(self, index, pattern):
        return -1 * self.nabla * (
                    self.map_vecindad(index).transpose() * (self.weights_map - pattern).transpose()).transpose()

    def map_vecindad(self, index):
        return np.exp(-(self.map_distances(index) ** 2) / 2 / self.sigma)

    def map_distances(self, index):
        return np.linalg.norm(self.source_map - self.source_map[index], axis=-1)

    def update_coeff(self):
        self.sigma *= self.step_coeff
        self.nabla *= self.step_coeff

    def init_coeffs(self, n):
        self.step_coeff = self.final_sigma ** (1 / (len(self.target) * n))
        self.sigma = self.start_sigma
        self.nabla = self.start_sigma

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
