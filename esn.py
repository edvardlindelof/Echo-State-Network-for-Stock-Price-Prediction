import numpy as np

class EchoStateNetwork():

    def __init__(self):
        self.wolo = "hej"

    def _rms_error(self, y, y_target):
        m = y.shape[0]
        return np.sqrt(np.sum((y - y_target) ** 2) / m)
