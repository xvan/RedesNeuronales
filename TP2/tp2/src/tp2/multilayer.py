import numpy as np
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
        trainer = MultilayerTrainer(data)
        trainer.train(self)


class MultilayerTrainer:

    def __init__(self, data: TrainDataType):
        self.learning_rate: float = 0.01
        self.iterations_limit: float = 100000
        self.data: TrainDataType = data
        self._init_exit_buffer()

    def _init_exit_buffer(self):
        buffer_size = 10
        self.last_costs = RingBuffer(buffer_size)
        self.long_costs = RingBuffer(buffer_size)

        for _ in range(buffer_size):
            self.last_costs.append(np.inf)
            self.long_costs.append(np.inf)

    def train(self, network: MultilayerNetwork):

        for perceptron in network.perceptrons:
            perceptron.weights = np.random.rand(*np.shape(perceptron.weights)) * 2 - 1

        best_weights = None
        best_cost = np.inf

        failed_attempts = 0
        while failed_attempts < 10:

            for _ in range(self.iterations_limit):
                np.random.shuffle(self.data)
                cost = 0
                for xo, y in self.data:

                    # Set States
                    network.process(xo)

                    cost += network.perceptrons[-1].sample_cost(y)
                    # BackPropagate Deltas

                    network.perceptrons[-1].first_delta(y)

                    for idx in list(range(len(network.perceptrons))[-2::-1]):
                        propagated_delta = network.perceptrons[idx + 1].back_propagate_delta()
                        network.perceptrons[idx].hidden_delta(propagated_delta)

                    # update weights:

                    for perceptron in network.perceptrons:
                        perceptron.update_weights(self.learning_rate)

                print(cost)
                self.last_cost.append(cost)
                mean_cost = np.mean(self.last_cost)
                self.long_costs.append(mean_cost)
                if 0.99 * self.long_costs[0] < mean_cost:
                    # print("Final Cost: %i" % cost)
                    break

            # if cost > best_cost:
            #     failed_attempts += 1
            # else:
            #     failed_attempts = 0
            #     best_weights = np.copy(self.weights)
            #     best_cost = cost

            if cost <= 0.005:
                break

