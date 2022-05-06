import unittest
import numpy as np
import numpy.testing as npt

from tp2.perceptron import Perceptron, ThresholdUnit


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
        npt.assert_array_equal(expected, Perceptron().append_bias(test))

    def test_and_gate(self):
        data = [
            ([1, 1], [1]),
            ([1, 0], [0]),
            ([0, 1], [0]),
            ([1, 1], [0])
        ]
        perceptron = Perceptron()
        perceptron.train(data)

class TestThresholdUnits(unittest.TestCase):
    def test_activator(self):
        npt.assert_array_equal(np.array([1, 1, -1]), ThresholdUnit().activate(np.array([4, 0, -5])))

    def test_process(self):
        data=[
            ([1, 1, 1], [1]),
            ([1, 0, -1], [1]),
            ([1, -1, -1], [-1]),
        ]
        tu=ThresholdUnit()
        tu.train(data)

        for (x, y) in data:
            npt.assert_array_equal(y, tu.process(x))

        #npt.assert_array_equal(np.array([1, 1, -1]), ThresholdUnit().activate(np.array([4, 0, -5])))

if __name__ == '__main__':
    unittest.main()
