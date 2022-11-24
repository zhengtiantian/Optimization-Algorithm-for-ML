import math

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


delta = 0.001


def getDx(x, minibatch):
    x1 = [x[0] + delta, x[1]]
    return (f(x1, minibatch) - f(x, minibatch)) / delta


def getDy(x, minibatch):
    x1 = [x[0], x[1] + delta]
    return (f(x1, minibatch) - f(x, minibatch)) / delta


def consStepSize(x, minibatch, sums, *param):
    alpha = param[0]
    dx = getDx(x, minibatch)
    dy = getDy(x, minibatch)
    x[0] = x[0] - alpha * dx
    x[1] = x[1] - alpha * dy
    return x, sums


def polyak(x, minibatch, sums, *param):
    epsilon = param[0]
    numerator = f(x, minibatch) - 0
    dx = getDx(x, minibatch)
    dy = getDy(x, minibatch)
    alpha = numerator / (dx ** 2 + dy ** 2 + epsilon)
    x[0] = x[0] - alpha * dx
    x[1] = x[1] - alpha * dy
    return x, sums


def RMSProp(x, minibatch, sums, *param):
    alpha0 = param[0]
    beta = param[1]
    epsilon = param[2]
    dx = getDx(x, minibatch)
    dy = getDy(x, minibatch)
    sums[0] = beta * sum[0] + (1 - beta) * (dx ** 2)
    sums[1] = beta * sum[1] + (1 - beta) * (dy ** 2)
    alpha = alpha0 / (math.sqrt(sums[0]) + epsilon)
    alpha2 = alpha0 / (math.sqrt(sums[1]) + epsilon)
    x[0] = x[0] - alpha * dx
    x[1] = x[1] - alpha2 * dy
    return x, sums


def heavyBall(x, minibatch, sums, *param):
    alpha = param[0]
    beta = param[1]
    dx = getDx(x, minibatch)
    dy = getDy(x, minibatch)
    sums[0] = beta * sums[0] + alpha * dx
    sums[1] = beta * sums[1] + alpha * dy
    x[0] = x[0] - sums[0]
    x[1] = x[1] - sums[1]
    return x, sums


def Adam(x, minibatch, sums, *param):
    alpha = param[0]
    beta1 = param[1]
    beta2 = param[1]
    epsilon = param[1]
    dx = getDx(x, minibatch)
    dy = getDy(x, minibatch)
    # mt
    sums[0] = beta1 * sums[0] + (1 - beta1) * dx
    # mt2
    sums[1] = beta1 * sums[1] + (1 - beta1) * dy
    # vt
    sums[2] = beta2 * sums[2] + (1 - beta2) * (dx ** 2)
    # vt2
    sums[3] = beta2 * sums[3] + (1 - beta2) * (dy ** 2)
    # i
    me = sums[0] / (1 - beta1 ** (sums[4] + 1))
    me2 = sums[1] / (1 - beta1 ** (sums[4] + 1))
    ve = sums[2] / (1 - beta2 ** (sums[4] + 1))
    ve2 = sums[3] / (1 - beta2 ** (sums[4] + 1))
    x[0] = x[0] - alpha * (me / (math.sqrt(ve) + epsilon))
    x[1] = x[1] - alpha * (me2 / (math.sqrt(ve2) + epsilon))
    sums[4] = sums[4] + 1


def miniBatchSGD(x, max_iter, batch_size, data, ii, xx, yy, ff, algo_func, *params):

    i = 0
    j = 0
    dataLen = len(data)
    sums = [0, 0, 0, 0, 0]
    while i < max_iter:
        np.random.shuffle(data)
        ii.append(i)
        xx.append(x[0])
        yy.append(x[1])
        ff.append(f(x, data))
        c_i = 0
        while c_i < dataLen:
            # ii.append(j)
            # xx.append(x[0])
            # yy.append(x[1])
            # ff.append(f(x, data))
            end = c_i + batch_size
            if end > dataLen:
                end = dataLen
            sample_data = data[c_i:end]
            x, sums = algo_func(x, sample_data, sums, *params)
            c_i = c_i + batch_size
            j = j + 1
        i = i + 1
    return ii, xx, yy, ff


data = generate_trainingdata()
# ii1, xx1, yy1, ff1 = miniBatchSGD([3, 3], 30, 2, data, [], [], [], [], consStepSize, 0.04)
# ii2, xx2, yy2, ff2 = miniBatchSGD([3, 3], 30, 5, data, [], [], [], [], consStepSize, 0.04)
# ii3, xx3, yy3, ff3 = miniBatchSGD([3, 3], 30, 10, data, [], [], [], [], consStepSize, 0.04)
ii1, xx1, yy1, ff1 = miniBatchSGD([3, 3], 50, 5, data, [], [], [], [], consStepSize, 0.01)
ii2, xx2, yy2, ff2 = miniBatchSGD([3, 3], 50, 5, data, [], [], [], [], consStepSize, 0.04)
ii3, xx3, yy3, ff3 = miniBatchSGD([3, 3], 50, 5, data, [], [], [], [], consStepSize, 0.1)
ii4, xx4, yy4, ff4 = miniBatchSGD([3, 3], 50, 5, data, [], [], [], [], consStepSize, 0.3)
ii5, xx5, yy5, ff5 = miniBatchSGD([3, 3], 50, 5, data, [], [], [], [], consStepSize, 0.9)
# ii4, xx4, yy4, ff4 = miniBatchSGD([3, 3], 30, 25, data, [], [], [], [], consStepSize, 0.5)
# ii5, xx5, yy5, ff5 = miniBatchSGD([3, 3], 30, 25, data, [], [], [], [], consStepSize, 0.9)
# ii3, xx3, yy3, ff3 = miniBatchSGD([3, 3], 50, 25, data, [], [], [], [], consStepSize, 0.5)
# ii4, xx4, yy4, ff4 = miniBatchSGD([3, 3], 50, 25, data, [], [], [], [], consStepSize, 0.9)
# ii3, xx3, yy3, ff3 = miniBatchSGD([3, 3], 50, 5, data, [], [], [], [], consStepSize, 0.1)
# ii4, xx4, yy4, ff4 = miniBatchSGD([3, 3], 50, 5, data, [], [], [], [], consStepSize, 0.3)
# ii5, xx5, yy5, ff5 = miniBatchSGD([3, 3], 50, 5, data, [], [], [], [], consStepSize, 0.5)
# ii6, xx6, yy6, ff6 = miniBatchSGD([3, 3], 50, 5, data, [], [], [], [], consStepSize, 0.9)
# ii1, xx1, yy1, ff1 = miniBatchSGD([3, 3], 50, 5, data, [], [], [], [], consStepSize, 0.1)
# ii2, xx2, yy2, ff2 = miniBatchSGD([3, 3], 50, 5, data, [], [], [], [], consStepSize, 0.3)
# ii3, xx3, yy3, ff3 = miniBatchSGD([3, 3], 50, 5, data, [], [], [], [], consStepSize, 0.9)
# ii4, xx4, yy4, ff4 = miniBatchSGD([3, 3], 50, 25, data, [], [], [], [], consStepSize, 0.1)
# ii5, xx5, yy5, ff5 = miniBatchSGD([3, 3], 50, 25, data, [], [], [], [], consStepSize, 0.3)
# ii6, xx6, yy6, ff6 = miniBatchSGD([3, 3], 50, 25, data, [], [], [], [], consStepSize, 0.9)
# ii1, xx1, yy1, ff1 = miniBatchSGD([3, 3], 20, 2, data, [], [], [], [], consStepSize, 0.1)
# ii2, xx2, yy2, ff2 = miniBatchSGD([3, 3], 20, 5, data, [], [], [], [], consStepSize, 0.1)
# ii3, xx3, yy3, ff3 = miniBatchSGD([3, 3], 20, 10, data, [], [], [], [], consStepSize, 0.1)
# ii4, xx4, yy4, ff4 = miniBatchSGD([3, 3], 50, 2, data, [], [], [], [], consStepSize, 0.9)
# ii5, xx5, yy5, ff5 = miniBatchSGD([3, 3], 50, 5, data, [], [], [], [], consStepSize, 0.9)
# ii6, xx6, yy6, ff6 = miniBatchSGD([3, 3], 50, 10, data, [], [], [], [], consStepSize, 0.9)

x = np.arange(-15, 10, 0.5)
y = np.arange(-20, 15, 0.5)

X, Y = np.meshgrid(x, y)
Z = np.zeros((len(X), len(X[0])))
for i, v in enumerate(X):
    for j, v2 in enumerate(v):
        Z[i][j] = f((X[i][j], Y[i][j]), data)
plt.figure(figsize=(7, 7))
# plt.plot(xx1, yy1, color='b', label='step size = 0.04, mini-batch size = 2')
# plt.plot(xx2, yy2, color='r', label='step size = 0.04, mini-batch size = 5')
# plt.plot(xx3, yy3, color='g', label='step size = 0.04, mini-batch size = 10')
# plt.plot(xx4, yy4, color='m', label='step size = 0.04, mini-batch size = 25')
# plt.plot(xx5, yy5, color='m', label='step size = 0.3, mini-batch size = 5')
# plt.plot(xx5, yy5, color='c', label='step size = 0.9, mini-batch size = 5')
# plt.plot(ii1, ff1, color='b', label='step size = 0.01, mini-batch size = 5')
# plt.plot(ii2, ff2, color='r', label='step size = 0.04, mini-batch size = 5')
# plt.plot(ii3, ff3, color='g', label='step size = 0.1, mini-batch size = 5')
# plt.plot(ii4, ff4, color='m', label='step size = 0.3, mini-batch size = 5')
# plt.plot(ii5, ff5, color='c', label='step size = 0.9, mini-batch size = 5')
# plt.plot(xx3, yy3, color='m', label='step size = 0.5, mini-batch size = 25')
# plt.plot(xx4, yy4, color='g', label='step size = 0.9, mini-batch size = 25')
# plt.plot(ii1, ff1, color='b', label='step size = 0.04, mini-batch size = 2')
# plt.plot(ii2, ff2, color='b', label='step size = 0.04, mini-batch size = 5')
# plt.plot(ii3, ff3, color='r', label='step size = 0.04, mini-batch size = 10')
plt.plot(xx1, yy1, color='b', label='step size = 0.01, mini-batch size = 5')
plt.plot(xx2, yy2, color='r', label='step size = 0.04, mini-batch size = 5')
plt.plot(xx3, yy3, color='g', label='step size = 0.1, mini-batch size = 5')
plt.plot(xx4, yy4, color='m', label='step size = 0.3, mini-batch size = 5')
plt.plot(xx5, yy5, color='c', label='step size = 0.9, mini-batch size = 5')
# plt.plot(xx6, yy6, color='r', label='step size = 0.9, mini-batch size = 5')
# plt.plot(ii1, ff1, color='y', label='step size = 0.1, mini-batch size = 5')
# plt.plot(ii2, ff2, color='g', label='step size = 0.3, mini-batch size = 5')
# plt.plot(ii3, ff3, color='r', label='step size = 0.9, mini-batch size = 5')
# plt.plot(ii4, ff4, color='b', label='step size = 0.1, full data set')
# plt.plot(ii5, ff5, color='c', label='step size = 0.3, full data set')
# plt.plot(ii6, ff6, color='m', label='step size = 0.9, full data set')
# plt.plot(ii1, ff1, color='b', label='step size = 0.1, mini-batch size = 2')
# plt.plot(ii2, ff2, color='r', label='step size = 0.1, mini-batch size = 5')
# plt.plot(ii3, ff3, color='y', label='step size = 0.1, mini-batch size = 10')
# plt.plot(ii4, ff4, color='y', label='step size = 0.9, mini-batch size = 2')
# plt.plot(ii5, ff5, color='g', label='step size = 0.9, mini-batch size = 5')
# plt.plot(ii6, ff6, color='b', label='step size = 0.9, mini-batch size = 10')
# plt.plot(xx4, yy4, color='y', label='step size = 0.9, mini-batch size = 2')
# plt.plot(xx5, yy5, color='g', label='step size = 0.9, mini-batch size = 5')
# plt.plot(xx6, yy6, color='b', label='step size = 0.9, mini-batch size = 10')
contours = plt.contour(X, Y, Z, 20);
plt.clabel(contours, inline=True, fontsize=10)
# plt.xlabel('epoches')
# plt.xlabel('iterations')
# plt.ylabel('function value of log')
plt.xlabel('x1')
plt.ylabel('x2')
# plt.yscale("log")
# plt.xscale("log")
# plt.ylim((-0.5, 0.1))
plt.legend()
plt.show()
