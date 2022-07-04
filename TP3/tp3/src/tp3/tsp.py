import numpy as np
from tp3.kohonen import KohonenNetwork




class EuclideanTravelingSalesProblemGenerator:
    def generate(self, n):
        return np.random.rand(n, 2)


class TspAutomapper(KohonenNetwork):
    def __init__(self, n):
        super().__init__((1, n))

    def map_distances(self, index):
        max_delta = 0.5
        distance = super().map_distances(index)/self.shape[-1]
        return max_delta - np.abs(np.mod(np.abs(distance), 2 * max_delta) - max_delta)

    # def init_source_map(self):
    #     aux = np.linspace(0, 1, self.shape[-1], False).reshape(1, self.shape[-1])
    #     return np.dstack([aux, np.zeros(aux.shape)])

    def initialize_weights_map(self):
        complex_circle = np.exp(2j*np.pi * self.source_map[:,:,0]/self.shape[-1])
        self.weights_map = np.array([complex_circle.real, complex_circle.imag]).transpose().reshape(*self.weights_map_shape)



