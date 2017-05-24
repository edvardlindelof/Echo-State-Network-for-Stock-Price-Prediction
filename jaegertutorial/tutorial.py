import numpy as np
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt


data = np.genfromtxt("aggregated_data", delimiter=", ")
scaled_data = (data - data.mean(axis=0)) / (data.std(axis=0))  # TODO scaling?
U = scaled_data[:,2:]
y = scaled_data[:,1].reshape(-1, 1)
#y_fut = scaled_data[:,0].reshape(-1, 1)
#y = y_fut - y

n_test = 20
U_train = U[0:-n_test]
y_train = y[0:-n_test]
U_test = U[-n_test:-1]
y_test = y[-n_test:-1]

n_features = U.shape[1]

reservoir_size = 100  # 100 in cs229
density = 0.05  # 0.05 in cs229 (actually 1 - 0.05, typo??), each node should have 10ish neighbours says practicalESN
spectral_radius = 0.8  # 0.8 in cs229, jaeger says usually between 0.7 and 0.98

# jaeger p 30
W_0 = (2 * np.random.rand(reservoir_size, reservoir_size) - 1) * \
      (np.random.rand(reservoir_size, reservoir_size) < density)
W_1 = W_0 / np.max(np.abs(np.linalg.eig(W_0)[0]))
W = spectral_radius * W_1

# from this plot it should be clear that the node values are periodical and decay to zero
n_iterations = 49
x = 10 * (np.random.rand(reservoir_size).reshape(-1, 1) - 1)
X = x
for _ in range(n_iterations):
    x = np.dot(W, x)
    X = np.hstack([X, x])
#plt.plot(X[0:5,:].T)
#plt.show()

# cs229 just says "random", jaeger says think about the squashing then trial-and-error, practicalESN says something
W_in = 2 * np.random.rand(reservoir_size, n_features) - 1

# cs229 just says "random", jaeger says think about the squashing then trial-and-error, practicalESN says something
W_back = 2 * np.random.rand(reservoir_size).reshape(-1, 1) - 1

def x_next(W_in, u_next, W, x, W_back, d):
    return np.tanh(np.dot(W_in, u_next) + np.dot(W, x) + W_back * d).reshape(-1, 1)

n_washout = 25  # 25 in cs229

#
# TODO everything below here not very well double-checked
# probably fault with some (n-1)-index or confusion between d and y
# jaeger p 29-32
#

x = np.zeros(reservoir_size).reshape(-1, 1)  # zeros in jaeger
d_scalar = 0
for i in range(n_washout):
    u = U[i].reshape(-1, 1)
    x = x_next(W_in, u, W, x, W_back, d_scalar)
    d_scalar = y[i]

M = np.hstack([u.T, x.T, [y[n_washout-2]]])  # TODO check that y-index
T = np.array([d_scalar]).reshape(1, 1)

for i in range(n_washout+1, U_train.shape[0]-1):
    u = U[i].reshape(-1, 1)
    x = x_next(W_in, u, W, x, W_back, d_scalar)

    m = np.hstack([u.T, x.T, [d_scalar]])  # TODO d(n-1)?
    M = np.vstack([M, m])
    d_scalar = y[i]
    T = np.vstack([T, d_scalar])

W_out = np.dot(np.linalg.pinv(M), T).T

y_pred = np.zeros(y_test.shape[0]).reshape(-1, 1)
for i in range(y_test.shape[0]):
    u = U[i].reshape(-1, 1)
    x = x_next(W_in, u, W, x, W_back, d_scalar)
    d_scalar = y_test[i]
    m = np.hstack([u.T, x.T, [d_scalar]])  # TODO d(n-1)?
    y_pred[i] = np.sum(W_out * m)

plt.plot(y_test)
plt.plot(y_pred)
plt.show()
plt.plot(y_test * y_pred)
plt.show()
