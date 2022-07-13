import os
import unittest

import numpy as np

from tp3.kohonen import SquareDataGenerator, KohonenNetwork, KohonenClustering
from tp3.tp3 import plot_map_for_distribution, plot_map_for_tsp, plot_mesh


class MyTestCase(unittest.TestCase):
    def test_something(self):
        plot_map_for_tsp(10)
        #self.assertEqual(True, False)  # add assertion here

    def test_manana(self):
        data = np.genfromtxt("../data/datos_para_clustering.csv", delimiter=",")
        kc = KohonenClustering([10, 10])
        val = kc.process(data)
        print(val)
        self.assertEqual(len(data), val.sum())
        plot_mesh(kc)

if __name__ == '__main__':
    unittest.main()
