import numpy as np

class EchoStateNetwork():

    def __init__(self):
        self.wolo = "hej"

    def _rms_error(self, y, y_target):
        m = y.shape[0]
        return np.sqrt(np.sum((y - y_target) ** 2) / m)

    def _x_next_tilde(self, W_in, u, W, x):
        u_star = np.vstack([[1], u])  # 1 prepended for bias
        return np.tanh(np.dot(W_in, u_star) + np.dot(W, x))

    def _x_next(self, alpha, x, x_tilde):
        return (1 - alpha) * x + alpha * x_tilde

    def _y(self, W_out, u, x):  # assumes one-dimensional output
        x_star = np.vstack([[1], u, x])  # 1 prepended for bias
        return np.dot(W_out, x_star)[0,0]

