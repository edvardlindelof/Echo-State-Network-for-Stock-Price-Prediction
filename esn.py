import numpy as np
import matplotlib
matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt



class EchoStateNetwork():

    def __init__(self, reservoir_size=10, alpha=1, beta=0, first_column_amplifier = 2):

        self.reservoir_size = reservoir_size
        self.alpha = alpha
        self.beta = beta
        self.first_column_amplifier = first_column_amplifier

    def fit(self, U, y):
        y = y.reshape(-1, 1)

        W_in_size = (self.reservoir_size,1+U.shape[1])
        first_column_amplifier = np.random.rand(1) * self.first_column_amplifier

        sparsity_W_in = np.random.rand(W_in_size[0],W_in_size[1]) > 0.75
        self.W_in = (np.random.rand(W_in_size[0], W_in_size[1]) - 0.5) * sparsity_W_in

        """other_column_amplifier = np.random.rand(1, self.W_in.shape[1]-1) * 2
        oca_length = other_column_amplifier.shape[0]

        self.W_in[:,0] = self.W_in[:,0] * first_column_amplifier

        for i in np.arange(1, oca_length):
            self.W_in[:,i] = self.W_in[:,i]* other_column_amplifier[i]
        """

        self.W = (np.random.rand(self.reservoir_size, self.reservoir_size) - 0.5)
        sparsity_W = np.random.rand(self.W.shape[0],self.W.shape[1]) > 0.2
        self.W = self.W * sparsity_W

        ## Introduces Spectral radius tuning
        eig_W, eig_vector_W= np.linalg.eig(self.W)
        abs_eig_W = np.absolute(eig_W)
        max_W = abs_eig_W.max()
        self.W = self.W / max_W

        print("The spectral radius is %.3f, Should be less than 1, for stable network" % max_W)

        x = np.zeros(self.reservoir_size).reshape(-1, 1)  + 0# TODO initial value

        Z_width = 1 + U.shape[1] + self.reservoir_size
        Z_height = U.shape[0]
        Z = np.empty((Z_height, Z_width))
        for i in range(0, Z_height):
            u = U[i].reshape(-1, 1)
            x, z = self._training_iteration(self.W_in, u, self.W, x, self.alpha)
            Z[i] = z.reshape(1, -1)

        #self.W_out = np.linalg.lstsq(Z, y)[0].reshape(1, -1)


        #Ridge Regression__
        expr1 = np.dot(y.T,Z)
        expr2 = np.linalg.inv( np.dot(Z.T,Z) + self.beta * np.identity(np.shape(Z)[1]) )
        self.W_out = np.dot( expr1, expr2 )

        self._x_init = x
        self.Z_train = Z;


    def predict(self, U):
        x = self._x_init
        y = np.empty(U.shape[0]).reshape(-1, 1)
        for i in range(0, U.shape[0]):
            u = U[i].reshape(-1, 1)
            x, y[i] = self._prediction_iteration(self.W_in, u, self.W,
                                                 x
                                                 , self.alpha, self.W_out)
        return y

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
        return x_next, z

    def _prediction_iteration(self, W_in, u, W, x, alpha, W_out):
        x_tilde = self._x_tilde(W_in, u, W, x)
        x_next = self._x_next(alpha, x, x_tilde)
        z = self._z(u, x_next)
        y_scalar = self._y_scalar(W_out, z)
        return x, y_scalar
