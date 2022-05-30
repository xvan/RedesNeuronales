import itertools
import unittest
import numpy as np
import numpy.testing as npt
import io
import pickle
import base64

from multiprocessing import Pool

from tp2.perceptron import Perceptron, ThresholdUnit, TrainDataType, NonLinearUnit
from tp2.capacity_estimator import CapacityEstimator, IncrementalEstimator
from tp2.multilayer import MultilayerNetwork
from typing import Callable, Tuple, List



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

large_and_or_gate_table: TrainDataType = [
    ([ 1,  1,  1,  1], [ 1,  1]),
    ([ 1,  1,  1, -1], [-1,  1]),
    ([ 1,  1, -1,  1], [-1,  1]),
    ([ 1,  1, -1, -1], [-1,  1]),
    ([ 1, -1,  1,  1], [-1,  1]),
    ([ 1, -1,  1, -1], [-1,  1]),
    ([ 1, -1, -1,  1], [-1,  1]),
    ([ 1, -1, -1, -1], [-1,  1]),
    ([-1,  1,  1,  1], [-1,  1]),
    ([-1,  1,  1, -1], [-1,  1]),
    ([-1,  1, -1,  1], [-1,  1]),
    ([-1,  1, -1, -1], [-1,  1]),
    ([-1, -1,  1,  1], [-1,  1]),
    ([-1, -1,  1, -1], [-1,  1]),
    ([-1, -1, -1,  1], [-1,  1]),
    ([-1, -1, -1, -1], [-1, -1]),
]

xor_gate_table: TrainDataType = [
    ([1, 1], [-1]),
    ([1, -1], [1]),
    ([-1, 1], [1]),
    ([-1, -1], [-1])
]

large_xor_gate_table: TrainDataType = [
    ([ 1,  1,  1,  1], [-1]),
    ([ 1,  1,  1, -1], [-1]),
    ([ 1,  1, -1,  1], [-1]),
    ([ 1,  1, -1, -1], [-1]),
    ([ 1, -1,  1,  1], [-1]),
    ([ 1, -1,  1, -1], [-1]),
    ([ 1, -1, -1,  1], [-1]),
    ([ 1, -1, -1, -1], [ 1]),
    ([-1,  1,  1,  1], [-1]),
    ([-1,  1,  1, -1], [-1]),
    ([-1,  1, -1,  1], [-1]),
    ([-1,  1, -1, -1], [ 1]),
    ([-1, -1,  1,  1], [-1]),
    ([-1, -1,  1, -1], [ 1]),
    ([-1, -1, -1,  1], [ 1]),
    ([-1, -1, -1, -1], [-1]),
]

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

class TestNonLinearUnit(unittest.TestCase):
    def test_process_with_pretrained_weights(self):
        lu = NonLinearUnit()
        lu.weights = [[150], [100], [100]]
        for (x, y) in and_gate_table:
            npt.assert_array_equal(y, lu.process(x))

    def test_cost(self):
        self.calculate_cost([0, 100], [([1], [1]), ([-1], [-1])])
        self.calculate_cost([150, 100, 100], and_gate_table)

    def calculate_cost(self, weights: np.ndarray, data: TrainDataType):
        lu = NonLinearUnit()
        lu.weights = weights
        self.assertGreater(0.001, lu.cost(data))
        pass

    def test_train_and_process(self):
        self.train_and_process(and_gate_table)
        self.train_and_process(and_or_gate_table)
        self.train_and_process(large_and_or_gate_table)
        self.train_and_process(sign_of_sum)

    def train_and_process(self, data: TrainDataType):
        gu = NonLinearUnit()
        gu.train(data)
        for (x, y) in data:
            npt.assert_array_equal(np.sign(y), np.sign(gu.process(x)))

class TestThresholdUnits(unittest.TestCase):
    def test_activator(self):
        npt.assert_array_equal(np.array([1, 1, -1]), ThresholdUnit().activate(np.array([4, 0, -5])))

    def test_process_with_pretrained_weights(self):
        tu = ThresholdUnit()
        tu.weights = [[1.5], [1], [1]]
        for (x, y) in and_gate_table:
            npt.assert_array_equal(y, tu.process(x))

    def test_train_and_process(self):
        self.train_and_process(and_gate_table)
        self.train_and_process(and_or_gate_table)
        self.train_and_process(sign_of_sum)

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

    pickleData="""
        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"""

    def decode_base64_picke(self, base64_str: str):
        with io.BytesIO(base64.b64decode(self.pickleData)) as fi:
            return pickle.load(fi)

    def test_train_and_process_difficult_pairs(self):
        error_samples = self.decode_base64_picke(self.pickleData)
        print(error_samples)
        for sample in error_samples:
            tu = ThresholdUnit()
            tu.train(sample, iterations_limit=10000)

class TestCapacityEstimator(unittest.TestCase):

    def setUp(self):
        self.ce = CapacityEstimator(ThresholdUnit())

    def test_capacity_by_bruteforce(self):
        self.assertEqual(1, self.ce.brute_capacity(1, 1))
        self.assertEqual(1, self.ce.brute_capacity(1, 2))
        self.assertEqual(1, self.ce.brute_capacity(2, 1))
        self.assertEqual(1, self.ce.brute_capacity(2, 2))
        self.assertEqual(1, self.ce.brute_capacity(2, 3))
        self.assertEqual(7 / 8, self.ce.brute_capacity(2, 4))
        # self.assertEqual(1, self.ce.capacity(3, 1))
        # self.assertEqual(1, self.ce.capacity(3, 2))
        # self.assertEqual(1, self.ce.capacity(3, 3))
        # self.assertEqual(137/140, self.ce.capacity(3, 4))
        # self.assertEqual(99/112, self.ce.capacity(3, 5))
        # self.assertEqual(41/56, self.ce.capacity(3, 6))
        # self.assertEqual(9/16, self.ce.capacity(3, 7))
        # self.assertEqual(13/32, self.ce.capacity(3, 8))

    def test_generate_data(self):
        ce = CapacityEstimator(ThresholdUnit())
        self.validate_generated_data(entries=3, patterns=2)
        self.validate_generated_data(entries=4, patterns=5)
        self.validate_generated_data(entries=8, patterns=128)

    def validate_generated_data(self, entries, patterns):
        data = self.ce.generate_data(entries, patterns)
        self.assertEqual(patterns, len(data))
        for x, y in data:
            self.assertEqual(entries, len(x))
            self.assertTrue(y in [1, -1])

    def test_capacity_by_montecarlo(self):
        self.assertAlmostEqual(1, self.ce.capacity(1, 1, 1000), delta=0.02)
        self.assertAlmostEqual(1, self.ce.capacity(1, 2, 1000), delta=0.02)
        #self.assertAlmostEqual(self.theoric_estimation(3), self.ce.capacity(1, 3, 1000), delta=0.02)
        #self.assertAlmostEqual(self.theoric_estimation(4), self.ce.capacity(1, 4, 1000), delta=0.02)

    def test_theoric_capacity(self):
        self.assertAlmostEqual(1, self.theoric_estimation(1))
        self.assertAlmostEqual(1, self.theoric_estimation(2))
        self.assertAlmostEqual(6/8, self.theoric_estimation(3))

    def theoric_estimation(self, dim_x: int):
        return np.sum([1 for x in itertools.product([1, -1], repeat=dim_x) if self.sign_changes(x) <= 1]) / 2**dim_x

    def test_sign_changes(self):
        self.assertEqual(0, self.sign_changes([1, 1]))
        self.assertEqual(1, self.sign_changes([-1, 1]))
        self.assertEqual(1, self.sign_changes([-1, -1, 1]))
        self.assertEqual(2, self.sign_changes([-1, 1, -1]))
        self.assertEqual(2, self.sign_changes([1, -1, -1, 1]))
        self.assertEqual(3, self.sign_changes([1, -1, 1, -1]))

    @staticmethod
    def sign_changes(x):
        old_x = x[0]
        count = 0
        for xi in x:
            if xi != old_x:
                count += 1
            old_x = xi
        return count

class TestIncrementalEstimator(unittest.TestCase):
    def test_starts_with_nan(self):
        ie = IncrementalEstimator()
        self.assertTrue(np.isnan(ie.mean))
        self.assertTrue(np.isnan(ie.variance))
        self.assertEqual(0,ie.count)

    def test_mean_of_one(self):
        ie = IncrementalEstimator()
        ie.append(1)
        self.assertAlmostEqual(1, ie.mean)
        self.assertTrue(np.isnan(ie.variance))

    def test_mean(self):
        self.validate_estimators([1, 1])
        self.validate_estimators([1, 2])
        self.validate_estimators(np.random.rand(100))

    def validate_estimators(self, data):
        ie = IncrementalEstimator()
        ie.append_range(data)
        self.assertAlmostEqual(np.mean(data), ie.mean)
        self.assertAlmostEqual(np.var(data, ddof=1), ie.variance),

    def test_std(self):
        ie = IncrementalEstimator()
        self.assertTrue(np.isnan(ie.std))
        data = np.random.rand(100)
        ie.append_range(data)
        self.assertAlmostEqual(np.std(data, ddof=1), ie.std)

class TestMultiLayerNetwork(unittest.TestCase):
    def test_has_correct_output_size(self):
        layers = [4, 2, 2]
        mn = MultilayerNetwork(layers)
        self.assertEqual(layers[-1], np.size(mn.process(np.ones(layers[0]))))

    def test_back_propagation(self):
        layers = [5, 3, 2]
        mn = MultilayerNetwork(layers)

    def test_train_and_process_xor(self):
        layers = [2, 2, 1]
        mn = MultilayerNetwork(layers)

        mn.train(xor_gate_table, 1)
        for (x, y) in xor_gate_table:
            npt.assert_array_equal(y, np.sign(mn.process(x)))

    def test_train_and_process_xor_gradient(self):
        layers = [2, 2, 1]
        mn = MultilayerNetwork(layers)

        mn.train(xor_gate_table, len(xor_gate_table))
        for (x, y) in xor_gate_table:
            npt.assert_array_equal(y, np.sign(mn.process(x)))

    def test_train_and_process_large_xor(self):
        layers = [4, 5, 1]
        mn = MultilayerNetwork(layers)

        mn.train(large_xor_gate_table, 1)
        for (x, y) in large_xor_gate_table:
            npt.assert_array_equal(y, np.sign(mn.process(x)))

    def test_train_and_process_large_xor_gradient(self):
        layers = [4, 5, 1]
        mn = MultilayerNetwork(layers)
        mn.train(large_xor_gate_table, 2)
        for (x, y) in large_xor_gate_table:
            npt.assert_array_equal(y, np.sign(mn.process(x)))

    @staticmethod
    def xor_task(_):
            layers = [2, 2, 1]
            mn = MultilayerNetwork(layers)
            mn.train(xor_gate_table, 1)
            return np.all([y == np.sign(mn.process(x)) for (x, y) in xor_gate_table])

    @unittest.skip("Too Slow")
    def test_streess_xor(self):
        iterations = 1000

        with Pool(processes=6) as pool:  # run no more than 6 at a time
            TT = pool.map(self.xor_task, range(iterations))
            successes = np.sum(TT)
            self.assertEqual(iterations, successes)

if __name__ == '__main__':
    unittest.main()
