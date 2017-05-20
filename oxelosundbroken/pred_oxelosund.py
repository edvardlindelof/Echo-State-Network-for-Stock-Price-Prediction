import numpy as np
import matplotlib
matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from collections import deque

from esn import EchoStateNetwork

raw_data = np.genfromtxt('temp_oxelosund.csv', delimiter=';', comments=';;', skip_header=10)
temperatures = raw_data[:, 3]

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
data_arr_normed = (data_arr - data_arr.mean(axis=0)) / data_arr.std(axis=0)
data_arr = data_arr_normed
print data_arr_normed.max()
print data_arr_normed.min()
print np.abs(data_arr_normed).mean()
'''
plt.plot(np.arange(25), data_arr[:, 0])
plt.plot(np.arange(25), data_arr[:, 1])
plt.plot(np.arange(25), data_arr[:, 2])
plt.show()
'''

n_test = 100
X = data_arr[:, 0:-1]
y = data_arr[:, -1]

X_train = X[0:-n_test]
y_train = y[0:-n_test]
X_test = X[-n_test:-1]
y_test = y[-n_test:-1]


resolution = 5
alpha = np.linspace(0.001,1,num=resolution)
beta = np.linspace(0.001,1,num=resolution)
first_column_amplifier = np.linspace(0.001,5,num=resolution)

sweep_esn = [[[EchoStateNetwork(50, a, b, fca) for fca in first_column_amplifier]for b in beta]for a in alpha]

y_preds = [[[[]  for i in range(alpha.shape[0])] for j in range(beta.shape[0])] for k in range(first_column_amplifier.shape[0])]
 
pred_error = [[[0  for i in range(alpha.shape[0])] for j in range(beta.shape[0])] for k in range(first_column_amplifier.shape[0])]


y_test_length = y_test.shape[0]

for i in range(alpha.shape[0]):
    for j in range(beta.shape[0]):
        for k in range(first_column_amplifier.shape[0]):
            print ('alpha: %.3f, beta: %.3f, fca: %.3f' %(alpha[i], beta[j], first_column_amplifier[k]))
            sweep_esn[i][j][k].fit(X_train, y_train)
            
            print ('alpha: %.3f, beta: %.3f, fca: %.3f' %(i,j,k))
            y_preds[i][j][k] = sweep_esn[i][j][k].predict(X_test)
                
            pred_error[i][j][k] = sweep_esn[i][j][k]._rms_error(y_preds[i][j][k] ,y_test)
            sweep_esn[i][j][k] = []



alphaX, betaY = np.meshgrid(alpha, beta)
errorZ = pred_error[:][:][1]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xlen = alpha.shape[0]
ylen = beta.shape[0]
colortuple = ('r', 'b')
colors = np.empty(X.shape, dtype=str)
for y in range(ylen):
    for x in range(xlen):
        colors[x, y] = colortuple[(x + y) % len(colortuple)]
        
# Plot the surface with face colors taken from the array we made.
surf = ax.plot_surface(X, Y, Z, facecolors=colors, linewidth=0)

surf = ax.plot_surface(alphaX, betaY, errorZ, facecolors=colors,
                        linewidth=0, )

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

"""
plt.figure(1)

n = 1000
z_iterations =  np.arange(n).reshape(-1,1)



print esn.Z_train[1:n,-1].reshape(-1,1).shape
print z_iterations.shape

plt_x1 = plt.plot(z_iterations,esn.Z_train[0:n,-1].reshape(-1,1), label='X[0]' )
plt_x10 = plt.plot(z_iterations,esn.Z_train[0:n,-10].reshape(-1,1), label='X[10]' )
plt_x30 = plt.plot(z_iterations,esn.Z_train[0:n,-30].reshape(-1,1), label='X[30]' )




plt.figure(2)
plt_real, = plt.plot(np.arange(n_test-1), y_test, label='realValues')
plt_predict, = plt.plot(np.arange(n_test-1), y_pred, label='prediction')
plt.legend([plt_real, plt_predict])


plt.show()  """
