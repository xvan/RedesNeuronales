import numpy as np
from typing import Callable, Tuple, List
from dvg_ringbuffer import RingBuffer

TrainDataType = List[Tuple[np.ndarray, np.ndarray]]


class TooManyIterations(Exception):
    pass


class Perceptron:
    import numpy as np

    def __init__(self):
        self.activator: Callable[[np.ndarray], np.ndarray] = None
        self.x: np.ndarray = None
        self.h: np.ndarray = None
        self.v: np.ndarray = None
        self.weights: np.ndarray = None


    def process(self, xi: np.ndarray):
        self.x = self._append_bias(xi)
        return self._process_biased(self.x)

    def _process_biased(self, xi_biased: np.ndarray):
        self.h = self._weight_product(xi_biased)
        self.v = self.activate(self.h)
        return self.v

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
        self._generate_weights_from_dim(np.size(first_xi), np.size(first_hi))

    def _generate_weights_from_dim(self, inDim: int, outDim: int):
        self.weights = np.ones((inDim+1, outDim))
        self.weights[0] = 0

    def apply_bias(self, xi: np.ndarray) -> np.ndarray:
        return self._append_bias(xi).reshape((1, -1))

    def apply_bias_to_data(self, data):
        return [(self.apply_bias(x), y) for x, y in data]


class NonLinearUnit(Perceptron):
    def __init__(self, inDim: int = 1 , outDim: int = 1):
        super().__init__()
        self.train_data: TrainDataType = None
        self.activator = np.tanh
        self.activatorDiff = lambda x: 1 - np.tanh(x) ** 2
        self._generate_weights_from_dim(inDim, outDim)

        self.delta: np.ndarray = None

    def cost(self, data: TrainDataType):
        return self.biased_cost(self.apply_bias_to_data(data))

    def biased_cost(self, biased_data: TrainDataType):
        return np.sum([(y - self._process_biased(x)) ** 2 for x, y in biased_data])

    def train(self, data: TrainDataType, learning_rate: float = 0.1, iterations_limit: int = 100000):
        self.train_data = data
        self._generate_weights(data)
        biased_data = self.apply_bias_to_data(data)

        last_costs = RingBuffer(10)
        long_costs = RingBuffer(10)

        for _ in range(10):
            last_costs.append(np.inf)
            long_costs.append(np.inf)

        best_weights = self.weights
        best_cost = np.inf

        failed_attempts = 0
        while failed_attempts < 10:
            self.weights = np.random.rand(* np.shape(self.weights)) * 2 - 1
            for _ in range(iterations_limit):
                cost_gradient = np.sum([self.cost_gradient(xb, y) for xb, y in biased_data], axis=0)
                self.weights += learning_rate * cost_gradient

                cost = self.biased_cost(biased_data)
                last_costs.append(cost)
                mean_cost = np.mean(last_costs)
                long_costs.append(mean_cost)
                if 0.99 * long_costs[0] < mean_cost:
                    # print("Final Cost: %i" % cost)
                    break

            if cost == 0:
                return

            if cost > best_cost:
                failed_attempts += 1
            else:
                failed_attempts = 0
                best_weights = self.weights
                best_cost = cost

        self.weights = best_weights
        print("Best Cost: %i" % best_cost)

    def cost_gradient(self, x_biased, y):
        wx = self._weight_product(x_biased)
        return ((y - self.activator(wx)) * self.activatorDiff(wx)) * x_biased.transpose()

    def first_delta(self, y: np.ndarray) -> np.ndarray:
        return self.hidden_delta(y - self.v)

    def hidden_delta(self, propagated_delta: np.ndarray) -> np.ndarray:
        self.delta = propagated_delta * self.activatorDiff(self.h)
        return self.delta

    def back_propagate_delta(self):
        return np.dot(self.delta, self.weights[1:, :].transpose()) #slice bias

    def update_weights(self, learning_rate: float):
        self.weights += learning_rate * self.x.reshape(-1,1).dot(self.delta.reshape(1, -1))

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

        curated_data = self.apply_bias_to_data(data)

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



