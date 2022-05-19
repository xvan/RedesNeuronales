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

    def train(self, data: TrainDataType, learning_rate: float = 0.01):
        last_costs = RingBuffer(10)
        long_costs = RingBuffer(10)

        for _ in range(10):
            last_costs.append(np.inf)
            long_costs.append(np.inf)

        for _ in range(1000):
            np.random.shuffle(data)
            for xo, y in data:

                #Set States
                self.process(xo)

                #BackPropagate Deltas

                self.perceptrons[-1].first_delta(y)

                for idx in list(range(len(self.perceptrons))[-2::-1]):
                    propagated_delta = self.perceptrons[idx+1].back_propagate_delta()
                    self.perceptrons[idx].hidden_delta(propagated_delta)

                #update weights:

                for perceptron in self.perceptrons:
                    perceptron.update_weights(learning_rate)



                # self.perceptrons[-2].weights += np.reshape(
                #    learning_rate * cost_delta * self.perceptrons[-2].x.transpose(), self.perceptrons[-2].weights.shape)


