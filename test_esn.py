import unittest
import numpy as np

from esn import EchoStateNetwork

class TestEchoStateNetwork(unittest.TestCase):

    esn = EchoStateNetwork()

    def test_rms_error(self):
        y = np.array([1, 1.5, 0]).reshape(-1, 1)
        y_target = np.array([-1, -0.5, 0.5]).reshape(-1, 1)
        by_class = self.esn._rms_error(y, y_target)
        by_hand = 1.6583123951776999
        self.assertEqual(by_class, by_hand)

if __name__ == '__main__':
    unittest.main()
