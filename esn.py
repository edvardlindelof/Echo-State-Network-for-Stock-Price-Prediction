import numpy as np

class EchoStateNetwork():

    def __init__(self, reservoir_size=10):
        self.reservoir_size = reservoir_size

    def fit(self, U, y):
        y = y.reshape(-1, 1)

        # TODO initial values
        self.W_in = np.random.rand(U.shape[1] + 1) - 0.5
        self.W = np.random.rand(self.reservoir_size, self.reservoir_size) - 0.5
        self.alpha = 0.5

        x = np.zeros(self.reservoir_size).reshape(-1, 1)  # TODO initial value
        Z_width = 1 + U.shape[1] + self.reservoir_size
        Z_height = U.shape[0]
        Z = np.empty((Z_height, Z_width))
        for i in range(0, Z_height):
            u = U[i].reshape(-1, 1)
            x, z = self._training_iteration(self.W_in, u, self.W, x, self.alpha)
            Z[i] = z.reshape(1, -1)

        self.W_out = np.linalg.lstsq(Z, y)[0].reshape(1, -1)

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
