import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from collections import deque

from esn import EchoStateNetwork

##
## USING JUST THE ORIGINAL FEATURES
##

raw_data = np.genfromtxt('SML_data_1', delimiter=' ', skip_header=1)

raw_data = raw_data[:, 2:]  # first two columns non-numeric

data_list = raw_data
data_arr = np.array(data_list)
print(data_arr.std(axis=0))
data_arr_normed = (data_arr - data_arr.mean(axis=0)) / (data_arr.std(axis=0) + 0.001)  # + 0.001 is ugly haxx
data_arr = data_arr_normed

n_test = 100
X = data_arr[:, 1:]
y = data_arr[:, 0]

X_train = X[0:-n_test]
y_train = y[0:-n_test]
X_test = X[-n_test:-1]
y_test = y[-n_test:-1]


resolution = 5
alpha = 0.5
beta = 1e1
first_column_amplifier = 1

esn = EchoStateNetwork(200, alpha, beta, first_column_amplifier)
esn.fit(X_train, y_train)
y_pred = esn.predict(X_test)

Z_train = esn.Z_train

plt.plot(y_test)
plt.plot(y_pred)
#plt.plot(Z_train[0:300, 30:35])
plt.show()

##
## USING AGGREGATED FEATURES AS WELL
##

raw_data = np.genfromtxt('SML_data_1_aggregated', delimiter=', ')

data_list = raw_data
data_arr = np.array(data_list)
print(data_arr.std(axis=0))
data_arr_normed = (data_arr - data_arr.mean(axis=0)) / (data_arr.std(axis=0) + 0.001)  # + 0.001 is ugly haxx
data_arr = data_arr_normed

n_test = 100
X = data_arr[:, 1:]
y = data_arr[:, 0]

X_train = X[0:-n_test]
y_train = y[0:-n_test]
X_test = X[-n_test:-1]
y_test = y[-n_test:-1]


resolution = 5
alpha = 0.5
beta = 1e1
first_column_amplifier = 1

esn = EchoStateNetwork(200, alpha, beta, first_column_amplifier)
esn.fit(X_train, y_train)
y_pred = esn.predict(X_test)

Z_train = esn.Z_train

plt.plot(y_test)
plt.plot(y_pred)
#plt.plot(Z_train[0:300, 30:35])
plt.show()


##
## USING AGGREGATED FEATURES, RIDGE REGRESSION (no reservoir)
##

raw_data = np.genfromtxt('SML_data_1_aggregated', delimiter=', ')

data_list = raw_data
data_arr = np.array(data_list)
print(data_arr.std(axis=0))
data_arr_normed = (data_arr - data_arr.mean(axis=0)) / (data_arr.std(axis=0) + 0.001)  # + 0.001 is ugly haxx
data_arr = data_arr_normed

n_test = 100
X = data_arr[:, 1:]
y = data_arr[:, 0]

X_train = X[0:-n_test]
y_train = y[0:-n_test]
X_test = X[-n_test:-1]
y_test = y[-n_test:-1]


resolution = 5
alpha = 0.5
beta = 1e1
first_column_amplifier = 1

# parameters should in practice be like having no reservoir at all
esn = EchoStateNetwork(2, alpha, beta, first_column_amplifier, density_W=1.)
esn.fit(X_train, y_train)
y_pred = esn.predict(X_test)

Z_train = esn.Z_train

plt.plot(y_test)
plt.plot(y_pred)
plt.show()
