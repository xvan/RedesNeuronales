import numpy as np
from tp3.kohonen import KohonenNetwork




class EuclideanTravelingSalesProblemGenerator:
    def generate(self, n):
        return np.random.rand(n, 2)


class TspAutomapper(KohonenNetwork):
    def __init__(self, n):
        super().__init__((n,))

    def map_distances(self, index):
        return self.circular_difference(self.source_map, self.source_map[index], 0.5)

    @staticmethod
    def circular_difference(a, b, max_delta):
        return max_delta - np.abs(np.mod(np.abs(a - b), 2 * max_delta) - max_delta)


