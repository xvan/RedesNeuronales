import unittest

from tp2.tp2 import add_one

class TestTp2(unittest.TestCase):

    def test_add_one(self):
        self.assertEqual(add_one(5), 6)


if __name__ == '__main__':
    unittest.main()