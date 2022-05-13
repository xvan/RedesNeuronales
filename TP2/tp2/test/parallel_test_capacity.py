import unittest
import numpy as np
import pandas as pd
import pickle

from multiprocessing import Pool

from tp2.perceptron import Perceptron, ThresholdUnit, TrainDataType
from tp2.capacity_estimator import CapacityEstimator, IncrementalEstimator

class ParallelTestCapacity(unittest.TestCase):


    def parallel_estimation(self):
        list(reversed(range(1, 11)))
        pool = Pool(processes=6)  # run no more than 6 at a time
        TT = pool.map(self.generate_estimates, reversed(range(1,11)))
        results = np.concatenate(TT)
        results_df = pd.DataFrame(results, columns=["dim", "pmax", "cap"])
        results_df.to_csv("~/results.csv")
        with open("/home/xvan/results.pkl", "wb") as fo:
             pickle.dump(results_df, fo)

    @staticmethod
    def generate_estimates(dim):
        results = []

        pmax = 1
        cap = 1
        while cap > 0:
            ce = CapacityEstimator(ThresholdUnit())
            cap = ce.capacity(dim, pmax, iterations_limit=1000)
            results.append([dim, pmax, cap])
            pmax += 1

        return results

if __name__ == '__main__':
    unittest.main()
