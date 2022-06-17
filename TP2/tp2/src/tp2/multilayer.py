from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.stats import norm
import copy
import random
from typing import Callable, List

from dvg_ringbuffer import RingBuffer

from tp2.perceptron import NonLinearUnit, TrainDataType
from tp2.capacity_estimator import IncrementalEstimator


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
        BackPropagationMultistartTrainer(self, data, chunk_size).train()


class InvalidChunkSize(Exception):
    pass


class AbstractMultilayerTrainer(metaclass=ABCMeta):
    def __init__(self, network: MultilayerNetwork, data: TrainDataType):
        self.network = network
        self.data: TrainDataType = copy.deepcopy(data)
        self.iterations_limit: float = 100000

    def set_weights(self, weights):
        for w, p in zip(weights, self.network.perceptrons):
            p.weights = w

    @abstractmethod
    def train(self):
        pass

    def extract_weights(self):
        return [np.copy(p.weights) for p in self.network.perceptrons]

    def generate_random_weights(self):
        return [np.random.rand(*np.shape(perceptron.weights)) * 2 - 1 for perceptron in self.network.perceptrons]

    def _random_initialize_weights(self):
        self.set_weights(self.generate_random_weights())

    def _set_network_states(self, xo, y):
        self.network.process(xo)
        return self.network.perceptrons[-1].sample_cost(y)

    def process_costs(self):
        return [self._set_network_states(x, y) for x, y in self.data]


class BackPropagationMultistartTrainer(AbstractMultilayerTrainer):
    def __init__(self, network: MultilayerNetwork, data: TrainDataType, chunk_size: int):
        if len(data) < chunk_size:
            raise InvalidChunkSize()

        super().__init__(network, data)

        self.best_weights = self.extract_weights()
        self.learning_rate: float = 0.01
        self.cost_target: float = 0.005
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
            if self._cost_target_reached:
                return
        self._restore_best_weights()

    @property
    def _cost_target_reached(self):
        return self.last_cost <= self.cost_target

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
            if self._improvement_not_significant() or self._cost_target_reached:
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

    def _improvement_not_significant(self):
        return 0.999 * self.old_mean_cost < self.mean_cost

    def _update_exit_control(self):
        if self.last_cost > self.best_cost:
            self.failed_attempts += 1
        else:
            self.failed_attempts = 0
            self.best_cost = self.last_cost
            self._save_best_weights()

    def _save_best_weights(self):
        self.best_weights = self.extract_weights()

    def _restore_best_weights(self):
        self.set_weights(self.best_weights)


class BackPropagationTrainer(BackPropagationMultistartTrainer):
    def __init__(self, network: MultilayerNetwork, data: TrainDataType, chunk_size: int):
        super().__init__(network, data, chunk_size)

    def _update_exit_control(self):
        self.failed_attempts = np.inf
        self.best_cost = self.last_cost
        self._save_best_weights()

    def _improvement_not_significant(self):
        return False


class SimulatedAnnealingDut(metaclass=ABCMeta):
    @abstractmethod
    def save_state(self):
        pass

    @abstractmethod
    def update_state(self):
        pass

    @abstractmethod
    def calculate_energy(self) -> float:
        pass

    @abstractmethod
    def rollback_state(self):
        pass


class SimulatedAnnealingStep:
    def __init__(self, dut: SimulatedAnnealingDut, temperature: float, confidence: float = 0.95, precision: float = 0.01):
        self.energy_target: float = 0.005
        self.dut = dut
        self._energy_estimators = IncrementalEstimator()
        self.percent_point = self._confidence_to_percent_point(confidence)
        self.precision = precision
        self.current_energy = np.inf
        self.temperature = temperature

    class InvalidConfidence(Exception):
        pass

    @staticmethod
    def _confidence_to_percent_point(confidence: float):
        if not 0 < confidence < 1:
            raise SimulatedAnnealingStep.InvalidConfidence()
        return norm.ppf((1 + confidence) / 2)

    @property
    def _stability_threshold(self):
        return self.percent_point * self._energy_estimators.std / np.sqrt(self._energy_estimators.count)

    @property
    def _stability_target(self):
        return self._energy_estimators.mean * self.precision

    @property
    def _energy_stability_reached(self):
        #print(self._stability_threshold, "<", self._stability_target)
        return self._stability_threshold < self._stability_target

    @property
    def energy_target_reached(self):
        return self.current_energy <= self.energy_target

    def should_accept_change(self, new_energy: float):
        delta_energy = new_energy - self.current_energy
        if delta_energy <= 0:
            return True
        is_accepted = np.random.rand() < np.exp(-delta_energy / self.temperature)
        return is_accepted

    def execute(self):
        self.dut.save_state()
        self.current_energy = np.inf

        while True:
            self.dut.update_state()
            new_energy = self.dut.calculate_energy()

            if self.should_accept_change(new_energy):
                self.current_energy = new_energy
                self.dut.save_state()
            else:
                self.dut.rollback_state()

            self.update_energy_estimator()
            if self.energy_target_reached | self._energy_stability_reached:
                return

    def update_energy_estimator(self):
        self._energy_estimators.append(self.current_energy)


class SimulatedAnnealingTrainer(AbstractMultilayerTrainer):
    def __init__(self, network: MultilayerNetwork, data: TrainDataType):
        super().__init__(network, data)
        self.temperature_callback: Callable[[float, float], None] = lambda t,c: None

    def train(self):
        start_temperature = 10000
        temperature_factor = 0.9
        temperature = start_temperature / temperature_factor #compensate first step

        while True:
            temperature *= temperature_factor
            step = SimulatedAnnealingStep(MultilayerAnnealingDut(self), temperature)
            step.execute()
            self.temperature_callback(temperature, step.current_energy)
            if step.energy_target_reached:
                return


class MultilayerAnnealingDut(SimulatedAnnealingDut):
    def __init__(self, trainer: SimulatedAnnealingTrainer):
        self.trainer = trainer
        self.weights = None
        self.sigma = 0.01/3

    def save_state(self):
        self.weights = self.trainer.extract_weights()

    def update_state(self):
        updated_weights = [weight + np.random.randn(*weight.shape) * self.sigma for weight in self.weights]
        self.trainer.set_weights(updated_weights)

    def calculate_energy(self):
        return np.mean(self.trainer.process_costs())

    def rollback_state(self):
        self.trainer.set_weights(self.weights)


class GeneticTrainer(AbstractMultilayerTrainer):
    def __init__(self, network: MultilayerNetwork, data: TrainDataType):
        super().__init__(network, data)
        self.pool_size = 10
        self.cross_over_probability = 0.2
        self.mutation_probability = 0.2
        self.mutation_std = 0.01
        self.generation_fitness = None
        self.generation_weights = []
        self.wights_numel = self.calculate_weights_size()
        self.error_target: float = 0.01

    @property
    def reproduction_odds(self):
        return self.generation_fitness / np.sum(self.generation_fitness)

    def train(self):
        self.generate_seed()
        while True:
            self.generation_fitness = self.calculate_generation_fitness()
            print(1/max(self.generation_fitness))
            if 1/max(self.generation_fitness) <= self.error_target:
                break
            self.step_generation()
        self.set_weights(self.generation_weights[np.argmax(self.generation_fitness)])

    def step_generation(self):
        survivor_indices = self.choose_survivors()
        crossed_weights = self.cross_survivors(survivor_indices)
        self.generation_weights = self.mutate_crossed(crossed_weights)

    def mutate_crossed(self, crossed_weights):
        return [self.mutate(weight) for weight in crossed_weights]

    def choose_survivors(self):
        return random.choices(list(range(self.pool_size)), self.generation_fitness, k=self.pool_size)

    def generate_seed(self):
        self.generation_weights = [self.generate_random_weights() for _ in range(self.pool_size)]

    def calculate_generation_fitness(self):
        return [self.calculate_specimen_fitness(specimen_weight) for specimen_weight in self.generation_weights]

    def calculate_specimen_fitness(self, weight):
        self.set_weights(weight)
        return 1/np.sum(self.process_costs())

    def cross_survivors(self, survivor_indices):
        crossed_weights = []
        crossed_weights += self.extract_fittest_if_odd(survivor_indices)
        crossed_weights += [y for idxA, idxB in self.iterate_pairs(survivor_indices)
                            for y in self.cross_pair(self.generation_weights[idxA], self.generation_weights[idxB])]
        return crossed_weights

    def extract_fittest_if_odd(self, survivors):
        return [survivors.pop(self.fittest_arg(survivors)), ] if len(survivors) % 2 == 1 else []

    @staticmethod
    def iterate_pairs(iterable):
        return ((iterable[i], iterable[i+1]) for i in list(range(0, len(iterable), 2)))

    def fittest_arg(self, survivors):
        return np.argmax(self.generation_fitness[survivor_index] for survivor_index in survivors)

    def cross_pair(self, subject_a, subject_b):
        subject_a = copy.deepcopy(subject_a)
        subject_b = copy.deepcopy(subject_b)

        recombination = random.randint(0, np.ceil(self.wights_numel / 2))

        idx = 0
        for layer_a, layer_b in zip(subject_a, subject_b):
            for (ia, wa), (ib, wb) in zip(np.ndenumerate(layer_a), np.ndenumerate(layer_b)):
                if idx < recombination:
                    layer_a[ia], layer_b[ib] = wb, wa
                else:
                    return subject_a, subject_b
                idx += 1

    def calculate_weights_size(self):
        return sum(layer.size for layer in self.extract_weights())

    def mutate(self, weight):
        if not self.has_mutation(): return weight
        return [layer + np.random.randn(*layer.shape) * self.mutation_std for layer in weight]

    def has_mutation(self):
        return random.random() > self.mutation_probability
