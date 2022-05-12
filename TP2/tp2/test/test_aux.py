import unittest
from tp2.perceptron import ThresholdUnit, TrainDataType
import tp2.aux as tp2Aux


class TestAuxFunctions(unittest.TestCase):

    def setUp(self):
        and_truth_table: TrainDataType = [
            ((1, 1), [1]),
            ((1, -1), [-1]),
            ((-1, 1), [-1]),
            ((-1, -1), [-1])
        ]

        self.tu = ThresholdUnit()
        self.tu.train(and_truth_table)


    def test_data_printing(self):
        print(tp2Aux.train_data_to_df(self.tu.train_data))

    def test_weight_printing(self):
        print(tp2Aux.weights_to_df(self.tu))

    def test_plot_call(self):
        tp2Aux.plot_2d_tu(self.tu)

class TestAuxFunctions(unittest.TestCase):

    def setUp(self):
        and_truth_table: TrainDataType = [
            ((1, 1), [1]),
            ((1, -1), [-1]),
            ((-1, 1), [-1]),
            ((-1, -1), [-1])
        ]

        self.tu = ThresholdUnit()
        self.tu.train(and_truth_table)


    def test_data_printing(self):
        print(tp2Aux.train_data_to_df(self.tu.train_data))

    def test_weight_printing(self):
        print(tp2Aux.weights_to_df(self.tu))

    def test_plot_call(self):
        tp2Aux.plot_2d_tu(self.tu)