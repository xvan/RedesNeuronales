import unittest
from tp2.perceptron import ThresholdUnit, TrainDataType
from tp2.multilayer import MultilayerTrainer, MultilayerNetwork, SingleAttemptMultilayerTrainer
import tp2.aux as tp2Aux

import numpy as np
import numpy.testing as npt

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

    def test_plot_all_costs(self):
        xor_gate_table = [
            ([1, 1], [-1]),
            ([1, -1], [1]),
            ([-1, 1], [1]),
            ([-1, -1], [-1])
        ]

        mn_xor2 = MultilayerNetwork([2, 2, 1])
        trainer_xor2 = MultilayerTrainer(mn_xor2, xor_gate_table, 1)
        trainer_xor2.train()
        tp2Aux.plot_all_cuts(trainer_xor2, xor_gate_table)




