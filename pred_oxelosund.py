import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from collections import deque

raw_data = np.genfromtxt('temp_oxelosund.csv', delimiter=';', comments=';;', skip_header=10)
temperatures = raw_data[0:30, 3]

data_list = []
dq = deque(maxlen=6)
for temp in temperatures:
    # dq.append(float(line[-1]))
    dq.append(temp)
    if len(dq) == 6:
        dql = list(dq)
        data_list.append([
            dql[-2],
            sum(dql[-6:-1]) / 5., # 5 day rolling avg
            dql[-1] # next days temp
        ])

data_arr = np.array(data_list)

plt.plot(np.arange(25), data_arr[:, 0])
plt.plot(np.arange(25), data_arr[:, 1])
plt.plot(np.arange(25), data_arr[:, 2])
plt.show()
