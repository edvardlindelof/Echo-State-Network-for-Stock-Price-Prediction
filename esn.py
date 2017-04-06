import numpy as np

class EchoStateNetwork():

    def __init__(self):
        self.wolo = "hej"

    def _rms_error(self, y, y_target):
        m = y.shape[0]
        return np.sqrt(np.sum((y - y_target) ** 2) / m)

    def _x_tilde(self, W_in, u, W, x):
        u_star = np.vstack([[1], u])  # 1 prepended for bias
        return np.tanh(np.dot(W_in, u_star) + np.dot(W, x))

    def _x_next(self, alpha, x, x_tilde):
        return (1 - alpha) * x + alpha * x_tilde

    def _z(self, u, x):
        return np.vstack([[1], u, x])  # 1 prepended for bias

    def _y_scalar(self, W_out, z):
        return np.dot(W_out, z)[0,0]

    def _training_iteration(self, W_in, u, W, x, alpha):
        x_tilde = self._x_tilde(W_in, u, W, x)
        x_next = self._x_next(alpha, x, x_tilde)
        z = self._z(u, x_next)
        return x, z

    def _prediction_iteration(self, W_in, u, W, x, alpha, W_out):
        x_tilde = self._x_tilde(W_in, u, W, x)
        x_next = self._x_next(alpha, x, x_tilde)
        z = self._z(u, x_next)
        y_scalar = self._y_scalar(W_out, z)
        return x, y_scalar
