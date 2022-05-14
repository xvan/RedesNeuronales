import numpy as np
from typing import Callable, Tuple, List

TrainDataType = List[Tuple[np.ndarray, np.ndarray]]


class TooManyIterations(Exception):
    pass


class Perceptron:
    import numpy as np

    def __init__(self):
        self.activator: Callable[[np.ndarray], np.ndarray] = None
        self.weights: np.ndarray = None

    def process(self, xi: np.ndarray):
        xi_biased = self._append_bias(xi)
        return self._process_biased(xi_biased)

    def _process_biased(self, xi_biased: np.ndarray):
        h = self._weight_product(xi_biased)
        return self.activate(h)

    def _weight_product(self, xi_biased: np.ndarray):
        return np.dot(xi_biased, self.weights)

    def activate(self, h: np.ndarray):
        return self.activator(h)

    def _append_bias(self, xi: np.ndarray) -> np.ndarray:
        return np.append(-1, xi)

    def train(self, data: TrainDataType, learning_rate: float = 0.01, iterations_limit: int = 1000):
        first_xi = data[0][0]
        self.weights = self._append_bias(np.ones(np.shape(first_xi))).transpose()

    def _generate_weights(self, data: np.ndarray):
        first_xi = data[0][0]
        first_hi = data[0][1]
        self.weights = np.ones((np.size(self._append_bias(first_xi)), np.size(first_hi)))
        self.weights[0] = 0

    def apply_bias(self, data):
        return [(self._append_bias(x).reshape((1, -1)), y) for x, y in data]


class LinearUnit(Perceptron):
    def __init__(self):
        super().__init__()
        self.train_data: TrainDataType = None
        self.activator = np.tanh
        self.activatorDiff = lambda x: 1 - np.tanh(x) ** 2

    def cost(self, data: TrainDataType):
        return self.biased_cost(self.apply_bias(data))

    def biased_cost(self, biased_data: TrainDataType):
        return np.sum([(y - self._process_biased(x)) ** 2 for x, y in biased_data])

    def train(self, data: TrainDataType, learning_rate: float = 1, iterations_limit: int = 100000):
        self.train_data = data
        self._generate_weights(data)
        biased_data = self.apply_bias(data)

        for _ in range(iterations_limit):
            cost = self.biased_cost(biased_data)
            print(cost, self.weights.transpose())

            if cost < 0.01:
                return

            cost_gradient = np.sum([self.cost_gradient(xb, y) for xb, y in biased_data])
            self.weights += learning_rate * cost_gradient


        raise TooManyIterations()

    def cost_gradient(self, x_biased, y):
        wx = self._weight_product(x_biased)
        return ((y - self.activator(wx)) * self.activatorDiff(wx)) * x_biased


class ThresholdUnit(Perceptron):

    @staticmethod
    def sgn(xi: np.ndarray) -> np.ndarray:
        return (xi >= 0).astype(int) * 2 - 1

    def __init__(self):
        super().__init__()
        self.train_data: TrainDataType = None
        self.activator = self.sgn

    def train(self, data: TrainDataType, learning_rate: float = 0.01, iterations_limit: int = 1000):
        self.train_data = data
        self._generate_weights(data)

        curated_data = self.apply_bias(data)

        clear = 0

        for _ in range(iterations_limit):

            for xb, err in ((xb, self._process_biased(xb) - y) for xb, y in curated_data):
                if np.any(err != 0):
                    self.weights -= learning_rate * np.dot(xb.transpose(), err)
                    clear = 0
                    continue

                clear += 1
                if clear == len(data):
                    return

        raise TooManyIterations()

    def log_step(self):
        print(self.weights.transpose())
        print([(x, y, self.process(x)) for x, y in self.train_data])



