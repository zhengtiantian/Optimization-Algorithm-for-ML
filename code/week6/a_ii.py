import numpy
import numpy as np
from matplotlib import pyplot as plt


def generate_trainingdata(m=25):
    return np.array([0, 0]) + 0.25 * np.random.randn(m, 2)


def f(x, minibatch):
    # loss function sum_{w in training data} f(x,w)
    y = 0;
    count = 0
    for w in minibatch:
        z = x - w - 1
        y = y + min(28 * (z[0] ** 2 + z[1] ** 2), (z[0] + 5) ** 2 + (z[1] + 9) ** 2)
        count = count + 1
    return y / count


data = generate_trainingdata()
x = np.arange(-12, 5, 0.5)
y = np.arange(-12, 5, 0.5)

X, Y = np.meshgrid(x, y)
Z = np.zeros((len(X), len(X[0])))
for i, v in enumerate(X):
    for j, v2 in enumerate(v):
        Z[i][j] = f((X[i][j], Y[i][j]), data)

fig = plt.figure(figsize=(7, 7))
contours = plt.contour(X, Y, Z, 20);
plt.clabel(contours, inline=True, fontsize=10)
plt.xlabel('x1')
plt.ylabel('x2')
# ax = fig.add_subplot(projection='3d')
# ax.plot_wireframe(X, Y, Z, color='b', label='')
# ax.set_xlabel('x1')
# ax.set_ylabel('x2')
# ax.set_zlabel('function value')
# plt.yscale("log")
# plt.xscale("log")
# plt.ylim((-0.5, 0.1))
plt.legend()
plt.show()
