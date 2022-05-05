import unittest
import numpy as np
import numpy.testing as npt

from tp2.perceptron import Perceptron


class TestPerceptron(unittest.TestCase):
    def activator(self):
        return 1

    perceptron = Perceptron(activator)

    def test_PerceptronProcess(self):
        output = self.perceptron.process(np.array([1]))

    def test_Process_Throws_TypeError_on_bad_input(self):
        bad_inputs = [None, "sorongo"]
        for bad_input in bad_inputs:
            with self.assertRaises(TypeError):
                print(bad_input)
                self.perceptron.process(bad_input)

    def test_bias(self):
        test = [1, 2, 3]
        expected = [-1, 1, 2, 3]
        npt.assert_array_equal(expected, self.perceptron.append_bias(test))


if __name__ == '__main__':
    unittest.main()
