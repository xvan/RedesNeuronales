import numpy as np
from typing import Callable, Tuple, List


class Perceptron:
    import numpy as np

    def __init__(self):
        self.activator: Callable[[np.ndarray], np.ndarray] = None
        self.weights: np.ndarray = None

    def process(self, xi: np.ndarray):
        xi_biased = self.append_bias(xi)
        h = np.dot(self.weights, xi_biased)
        return self.activate(h)

    def activate(self, h: np.ndarray):
        return self.activator(h)

    def append_bias(self, xi: np.ndarray) -> np.ndarray:
        return np.append(-1, xi)

    def train(self, data: List[Tuple[np.ndarray,np.ndarray]]):
        first_xi = data[0][0]
        self.weights = self.append_bias(np.ones(np.shape(first_xi))).transpose()



class ThresholdUnit(Perceptron):

    @staticmethod
    def sgn(xi: np.ndarray) -> np.ndarray:
        return (xi >= 0).astype(int) * 2 - 1

    def __init__(self):
        super().__init__()
        self.activator = self.sgn

    def train(self, data) -> np.ndarray:
        first_xi = data[0][0]
        self.weights = self.append_bias(np.ones(np.shape(first_xi))).transpose()
        self.weights[0]=0

