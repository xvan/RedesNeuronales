import numpy as np
from typing import Callable


class Perceptron:
    import numpy as np

    def __init__(self, activator):
        self.activator: Callable[[np.ndarray], np.ndarray] = activator
        self.weights: np.ndarray = np.array([1])

    def process(self, xi: np.ndarray):
        xi_fix = self.append_bias(xi)
        h = self.weights * xi_fix
        return self.activate(h)

    def activate(self, h: np.ndarray):
        return self.activator(h)

    def append_bias(self, xi: np.ndarray) -> np.ndarray:
        return np.append(-1, xi)
