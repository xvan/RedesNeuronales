import unittest

from tp3.kohonen import SquareDataGenerator
from tp3.tp3 import plot_map_for_distribution, plot_map_for_tsp


class MyTestCase(unittest.TestCase):
    def test_something(self):
        plot_map_for_tsp(10)
        #self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
