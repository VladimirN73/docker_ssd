import unittest
from simple import simpleA


class simpleTest(unittest.TestCase):
    def test_SimpleA(self):
        self.assertEqual(simpleA('hello'), 'hello')


if __name__ == '__main__':
    unittest.main()
