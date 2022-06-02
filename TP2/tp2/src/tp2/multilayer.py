import numpy as np
import copy
import itertools
import random
from typing import Callable, Tuple, List

from dvg_ringbuffer import RingBuffer

from tp2.perceptron import NonLinearUnit, TrainDataType


class MultilayerNetwork:
    def __init__(self, layers: List[int]):
        self.perceptrons = [NonLinearUnit(inDim, outDim) for inDim, outDim in zip(layers[:-1], layers[1:])]
        self.state = [[]] * len(layers)

    def process(self, xo: np.ndarray):
        i = 0
        self.state[i] = xo
        for perceptron in self.perceptrons:
            self.state[i + 1] = perceptron.process(self.state[i])
            i += 1
        return self.state[-1]

    def train(self, data: TrainDataType, chunk_size=1):
        MultilayerTrainer(self, data, chunk_size).train()


class InvalidChunkSize(Exception):
    pass


class MultilayerTrainer:
    def __init__(self, network: MultilayerNetwork, data: TrainDataType, chunk_size: int):
        if len(data) < chunk_size:
            raise InvalidChunkSize()

        self.network = network
        self.learning_rate: float = 0.01
        self.iterations_limit: float = 100000
        self.data: TrainDataType = copy.deepcopy(data)
        self.chunk_size = int(chunk_size)
        self.cost_callback: Callable[[float], None] = lambda c: None

    def _init_costs_log(self):
        last_buffer_size = 10
        long_buffer_size = 10
        self.last_costs = RingBuffer(last_buffer_size)
        self.long_costs = RingBuffer(long_buffer_size)

        for _ in range(last_buffer_size):
            self.last_costs.append(np.inf)

        for _ in range(long_buffer_size):
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

    @property
    def number_of_chunks(self) -> int:
        return int(np.floor(len(self.data) / self.chunk_size))

    @property
    def shuffled_data(self) -> TrainDataType:
        return random.sample(self.data, self.number_of_chunks * self.chunk_size)

    @property
    def chunked_data(self) -> List[TrainDataType]:
        data = np.array(self.shuffled_data, dtype=object)
        return np.array_split(data, self.number_of_chunks)

    def _train_step(self):
        chunks = self.chunked_data
        return np.sum([self._train_chunk(chunk) for chunk in chunks])

    def _train_chunk(self, chunk: TrainDataType) -> float:
        sample_costs, sample_deltas = zip(*[self._train_sample(xo, y) for xo, y in chunk])
        self._set_deltas(np.mean(sample_deltas, axis=0))
        self._update_weights()
        return np.mean(sample_costs)

    def _train_sample(self, xo, y):
        cost = self._set_network_states(xo, y)
        self._backpropagate_deltas(y)
        # self._save_deltas()
        # self._update_weights()
        return cost, self._get_deltas()

    def _set_deltas(self, deltas: List[List[float]]):
        for perceptron, delta in zip(self.network.perceptrons, deltas):
            perceptron.delta = delta

    def _update_weights(self):
        for perceptron in self.network.perceptrons:
            perceptron.update_weights(self.learning_rate)

    def _get_deltas(self):
        return [copy.deepcopy(perceptron.delta) for perceptron in self.network.perceptrons]

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


class SingleAttemptMultilayerTrainer(MultilayerTrainer):
    def __init__(self, network: MultilayerNetwork, data: TrainDataType, chunk_size: int):
        super().__init__(network, data, chunk_size)

    def _update_exit_control(self):
        self.failed_attempts = np.inf
        self.best_cost = self.last_cost
        self._save_best_weights()

    def _improvement_not_significant(self):
        return False

