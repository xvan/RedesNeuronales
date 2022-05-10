import unittest
import numpy as np
import numpy.testing as npt

from tp2.perceptron import Perceptron, ThresholdUnit, TrainDataType
from typing import Callable, Tuple, List

class TestPerceptron(unittest.TestCase):
    def activator(self, xi):
        return 1

    def test_Process_Throws_Without_Training(self):
        per = Perceptron()
        per.activator = lambda x: 1
        with self.assertRaises(TypeError):
            per.process(np.ndarray([1]))

    def test_Process_With_Training(self):
        per = Perceptron()
        per.activator = lambda x: 1
        per.train([([1],[1])])
        per.process([1])

    def test_Process_Throws_on_bad_input(self):
        bad_inputs = [None, "string"]
        for bad_input in bad_inputs:
            with self.assertRaises(TypeError):
                print(bad_input)
                Perceptron().process(bad_input)

    def test_bias(self):
        test = [1, 2, 3]
        expected = [-1, 1, 2, 3]
        biased = Perceptron()._append_bias(test)
        npt.assert_array_equal(expected, biased)

    and_gate_table: TrainDataType = [
        ([1, 1], [1]),
        ([1, 0], [0]),
        ([0, 1], [0]),
        ([1, 1], [0])
    ]

    trainingData2: TrainDataType = [
        ([1, 1, 1], [1]),
        ([1, 0, 1], [0]),
        ([0, 1, 1], [0]),
        ([1, 1, 1], [0])
    ]

    trainingData3: TrainDataType = [
        ([1, 1, 1], [1, 1]),
        ([1, 0, 1], [0, 1]),
        ([0, 1, 1], [0, 1]),
        ([1, 1, 1], [0, 1])
    ]

    def test_generate_weights(self):
        self.weights_have_the_right_size(self.and_gate_table, (3, 1))
        self.weights_have_the_right_size(self.trainingData2, (4, 1))
        self.weights_have_the_right_size(self.trainingData3, (4, 2))

    def weights_have_the_right_size(self, data: TrainDataType, size: Tuple[int, int]):
        pt = Perceptron()
        pt._generate_weights(data)
        npt.assert_array_equal(size, np.shape(pt.weights))

    def test_and_gate(self):
        perceptron = Perceptron()
        perceptron.train(self.and_gate_table)

class TestThresholdUnits(unittest.TestCase):
    def test_activator(self):
        npt.assert_array_equal(np.array([1, 1, -1]), ThresholdUnit().activate(np.array([4, 0, -5])))

    and_gate_table: TrainDataType = [
        ([1, 1], [1]),
        ([1, -1], [-1]),
        ([-1, 1], [-1]),
        ([-1, -1], [-1])
    ]

    and_or_gate_table: TrainDataType = [
        ([1, 1], [1, 1]),
        ([1, -1], [-1, 1]),
        ([-1, 1], [-1, 1]),
        ([-1, -1], [-1, -1])
    ]

    sign_of_sum: TrainDataType = [
        ([1, 1, 1], [1]),
        ([1, 0, -1], [1]),
        ([1, -1, -1], [-1]),
    ]

    xor_gate_table: TrainDataType = [
        ([1, 1], [-1]),
        ([1, -1], [1]),
        ([-1, 1], [1]),
        ([-1, -1], [-1])
    ]

    def test_process_with_pretrained_weights(self):
        tu = ThresholdUnit()
        tu.weights = [[1.5], [1], [1]]
        for (x, y) in self.and_gate_table:
            npt.assert_array_equal(y, tu.process(x))

    def test_train_and_process(self):
        self.train_and_process(self.and_gate_table)
        self.train_and_process(self.and_or_gate_table)
        self.train_and_process(self.sign_of_sum)

    def train_and_process(self, data: TrainDataType):
        tu = ThresholdUnit()
        tu.train(data)
        for (x, y) in data:
            npt.assert_array_equal(y, tu.process(x))

    def test_train_throws_on_non_linear_separable(self):
        bad_inputs = [None, "string"]
        for bad_input in bad_inputs:
            with self.assertRaises(TypeError):
                print(bad_input)
                Perceptron().process(bad_input)

if __name__ == '__main__':
    unittest.main()
