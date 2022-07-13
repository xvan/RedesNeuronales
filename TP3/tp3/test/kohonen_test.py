import unittest
import numpy as np
from tp3.kohonen import CircularDataGenerator, KohonenNetwork, SquareDataGenerator, KohonenClustering
from tp3.tsp import TspAutomapper


class KohonenTestCase(unittest.TestCase):
    def test_circular_generator(self):
        n_samples = 100
        generated_data = CircularDataGenerator().generate(n_samples)
        self.assertEqual((n_samples, 2), np.shape(generated_data))
        self.assertTrue(np.all(np.linalg.norm(generated_data, axis=1) <= 1))

    # def test_generate_map(self):
    #     kn = KohonenNetwork([5, 5])
    #     self.assertIsNone(kn.weights_map)
    #     kn.train(np.ones((1, 3)))
    #     self.assertEqual((5, 5, 3), kn.weights_map.shape)

    def test_run(self):
        target = CircularDataGenerator().generate(50)
        kn = KohonenNetwork([5, 5])
        kn.set_target(target)
        kn.train()


class TspTestCase(unittest.TestCase):
    def test_run(self):
        N = 10
        target = SquareDataGenerator().generate(N)
        ta = TspAutomapper(N)
        ta.set_target(target)
        ta.train(100)


class ClusteringCase(unittest.TestCase):
    def test_operations(self):
        test_map = np.dstack((np.arange(2)[:,None].repeat(3, axis=1),np.arange(3)[:, None].repeat(2, axis=1).transpose()))
        test_values = np.array([[0, 0], [.25, .25], [1, 1], [1.25, 1.25], [0, 2]])
        objective = [[2, 0, 1], [0, 2, 0]]
        evaluated = KohonenClustering.generate_clusters(test_map, test_values)
        self.assertTrue(np.all(evaluated == objective))

    def test_algorithm(self):
        kc = KohonenClustering([2, 3])
        test_values = np.array([[0, 0], [.25, .25], [1, 1], [1.25, 1.25], [0, 2]])
        val=kc.process(test_values)
        print(val)

    def test_matrix_U(self):
        kc = KohonenClustering([2, 3])
        test_values = np.array([[0, 0], [.25, .25], [1, 1], [1.25, 1.25], [0, 2]])
        kc.process(test_values)
        kc.mean_distances()

    def test_run(self):
        data = np.genfromtxt("../data/datos_para_clustering.csv", delimiter=",")
        kc = KohonenClustering([10, 10])
        val = kc.process(data)
        print(val)
        self.assertEqual(len(data), val.sum())


if __name__ == '__main__':
    unittest.main()
