import numpy as np
import copy
import itertools
from typing import Callable, Tuple, List

from dvg_ringbuffer import RingBuffer

from tp2.perceptron import NonLinearUnit, TrainDataType

class MultilayerNetwork:
    def __init__(self, layers: List[int]):
        self.perceptrons=[NonLinearUnit(inDim, outDim) for inDim, outDim in zip(layers[:-1], layers[1:])]
        self.state = [[]] * len(layers)

    def process(self, xo: np.ndarray):
        i = 0
        self.state[i] = xo
        for perceptron in self.perceptrons:
            self.state[i+1] = perceptron.process(self.state[i])
            i += 1
        return self.state[-1]

    def train(self, data: TrainDataType):
        MultilayerTrainer(self, data).train()


class MultilayerTrainer:
    def __init__(self, network: MultilayerNetwork, data: TrainDataType ):
        self.network = network
        self.learning_rate: float = 0.01
        self.iterations_limit: float = 100000
        self.data: TrainDataType = copy.deepcopy(data)
        self.cost_callback: Callable[[float], None] = lambda c: None

    def _init_costs_log(self):
        buffer_size = 10
        self.last_costs = RingBuffer(buffer_size)
        self.long_costs = RingBuffer(buffer_size)

        for _ in range(buffer_size):
            self.last_costs.append(np.inf)
            self.long_costs.append(np.inf)

    @property
    def last_cost(self):
        return self.last_costs[-1]

    @property
    def mean_cost(self):
        return self.long_costs[-1]

    @property
    def old_mean_cost(self):
        return self.long_costs[0]

    def train(self):
        self._init_exit_control()
        while self.failed_attempts < 10:
            self._train_attempt()
            self._update_exit_control()
            if self.last_cost <= 0.005:
                return
        self._restore_best_weights()

    def _restore_best_weights(self):
        for w, p in zip(self.best_weights, self.network.perceptrons):
            p.weights = w

    def _save_best_weights(self):
        self.best_weights = [np.copy(p.weights) for p in self.network.perceptrons]

    def _init_attempt_states(self):
        self._random_initialize_weights()
        self._init_costs_log()

    def _init_exit_control(self):
        self.best_weights = None
        self.best_cost = np.inf
        self.failed_attempts = 0

    def _train_attempt(self):
        self._init_attempt_states()
        for _ in range(self.iterations_limit):
            np.random.shuffle(self.data)
            cost = self._train_step()
            self._update_costs_log(cost)
            if self._improvement_not_significant():
                break

    def _update_costs_log(self, cost):
        # print(cost)
        self.last_costs.append(cost)
        mean_cost = np.mean(self.last_costs)
        self.long_costs.append(mean_cost)
        self.cost_callback(cost)

    def _train_step(self):
        return np.sum([self._train_sample(xo, y) for xo, y in self.data])

    def _train_sample(self, xo, y):
        cost = self._set_network_states(xo, y)
        self._backpropagate_deltas(y)
        self._update_weights()
        return cost

    def _update_weights(self):
        for perceptron in self.network.perceptrons:
            perceptron.update_weights(self.learning_rate)

    def _backpropagate_deltas(self, y):
        self.network.perceptrons[-1].first_delta(y)
        for idx in list(range(len(self.network.perceptrons))[-2::-1]):
            propagated_delta = self.network.perceptrons[idx + 1].back_propagate_delta()
            self.network.perceptrons[idx].hidden_delta(propagated_delta)

    def _set_network_states(self, xo, y):
        self.network.process(xo)
        return self.network.perceptrons[-1].sample_cost(y)

    def _random_initialize_weights(self):
        for perceptron in self.network.perceptrons:
            perceptron.weights = np.random.rand(*np.shape(perceptron.weights)) * 2 - 1

    def _improvement_not_significant(self):
        return 0.999 * self.old_mean_cost < self.mean_cost

    def _update_exit_control(self):
        if self.last_cost > self.best_cost:
            self.failed_attempts += 1
        else:
            self.failed_attempts = 0
            self.best_cost = self.last_cost
            self._save_best_weights()

