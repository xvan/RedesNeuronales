import unittest
import numpy as np
from tp3.kohonen import CircularDataGenerator, KohonenNetwork


class KohonenTestCase(unittest.TestCase):
    def test_circular_generator(self):
        n_samples = 100
        generated_data = CircularDataGenerator().generate(n_samples)
        self.assertEqual((n_samples, 2), np.shape(generated_data))
        self.assertTrue(np.all(np.linalg.norm(generated_data, axis=1) <= 1))

    def test_generate_map(self):
        kn = KohonenNetwork([5, 5])
        self.assertIsNone(kn.weights_map)
        kn.train(np.ones((1, 3)))
        self.assertEqual((5, 5, 3), kn.weights_map.shape)

    def test_run(self):
        target = CircularDataGenerator().generate(50)
        kn = KohonenNetwork([5, 5])
        kn.train(target)



if __name__ == '__main__':
    unittest.main()