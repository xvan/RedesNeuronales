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
        h = np.dot(xi_biased, self.weights)
        return self.activate(h)

    def activate(self, h: np.ndarray):
        return self.activator(h)

    def _append_bias(self, xi: np.ndarray) -> np.ndarray:
        return np.append(-1, xi)

    def train(self, data: TrainDataType):
        first_xi = data[0][0]
        self.weights = self._append_bias(np.ones(np.shape(first_xi))).transpose()

    def _generate_weights(self, data: np.ndarray):
        first_xi = data[0][0]
        first_hi = data[0][1]
        self.weights = np.ones((np.size(self._append_bias(first_xi)), np.size(first_hi)))
        self.weights[0] = 0

class ThresholdUnit(Perceptron):

    @staticmethod
    def sgn(xi: np.ndarray) -> np.ndarray:
        return (xi >= 0).astype(int) * 2 - 1

    def __init__(self):
        super().__init__()
        self.train_data: TrainDataType = None
        self.activator = self.sgn

    ITERATIONS_LIMIT = 1000

    def train(self, data: TrainDataType):
        self.train_data = data
        self._generate_weights(data)

        curated_data = [(self._append_bias(x).reshape((1, -1)), y) for x, y in data]

        clear = 0

        learning_rate = 0.01

        for _ in range(self.ITERATIONS_LIMIT):
            for xb, err in ((xb, self._process_biased(xb) - y) for xb, y in curated_data):
                if np.any(err != 0):
                    self.weights -= learning_rate * np.dot(xb.transpose(), err)
                    clear = 0
                    continue

                clear += 1
                if clear == len(data):
                    return

            if clear == len(data):
                return

        raise TooManyIterations()

    def log_step(self):
        print(self.weights.transpose())
        print([(x, y, self.process(x)) for x, y in self.train_data])



