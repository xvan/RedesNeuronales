import unittest
import numpy as np
import pandas as pd
import pickle

from tp2.perceptron import Perceptron, ThresholdUnit, TrainDataType
from tp2.capacity_estimator import CapacityEstimator, IncrementalEstimator

class TestCapacity(unittest.TestCase):
    def generate_estimates(self):
        results = []
        for dim in list(range(1, 11)):
            pmax = 1
            cap = 1
            while cap > 0:
                ce = CapacityEstimator(ThresholdUnit())
                cap = ce.capacity(dim, pmax, iterations_limit=1000)
                results.append([dim, pmax, cap])
                pmax += 1

        results_df = pd.DataFrame(results, columns=["dim", "pmax", "cap"])
        results_df.to_csv("~/results.csv")
        with open("/home/xvan/results.pkl", "wb") as fo:
            pickle.dump(results_df, fo)

if __name__ == '__main__':
    unittest.main()
