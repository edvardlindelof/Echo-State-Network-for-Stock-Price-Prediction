import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))  # lets us import from other dirs


import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge


data = np.genfromtxt('sca/aggregated_data', delimiter=', ')

data_arr_normed = (data - data.mean(axis=0)) / (data.std(axis=0))
data = data_arr_normed

n_test = 100
X = data[:, 1:]  # first column is next close (to be predicted)
y = data[:, 0] - X[:,0]  # next close minus last close

X_train = X[0:-n_test]
y_train = y[0:-n_test]
X_test = X[-n_test:-1]
y_test = y[-n_test:-1]

ridge = Ridge()
ridge.fit(X_train, y_train)
print(ridge.alpha)
print(ridge.coef_)
y_pred = ridge.predict(X_test)

plt.plot(y_test)
plt.plot(y_pred)
plt.show()
