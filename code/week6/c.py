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
    print(alpha)
    x[0] = x[0] - alpha * dx
    x[1] = x[1] - alpha * dy
    return x, sums


def RMSProp(x, minibatch, sums, *param):
    alpha0 = param[0]
    beta = param[1]
    epsilon = param[2]
    dx = getDx(x, minibatch)
    dy = getDy(x, minibatch)
    sums[0] = beta * sums[0] + (1 - beta) * (dx ** 2)
    sums[1] = beta * sums[1] + (1 - beta) * (dy ** 2)

    alpha = alpha0 / (math.sqrt(sums[0]) + epsilon)
    alpha2 = alpha0 / (math.sqrt(sums[1]) + epsilon)
    print(alpha)
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


def adam(x, minibatch, sums, *param):
    alpha = param[0]
    beta1 = param[1]
    beta2 = param[2]
    epsilon = param[3]
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
    print((me / (math.sqrt(ve) + epsilon)))
    x[1] = x[1] - alpha * (me2 / (math.sqrt(ve2) + epsilon))
    sums[4] = sums[4] + 1
    return x, sums


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

ii, xx, yy, ff = miniBatchSGD([3, 3], 50, 5, data, [], [], [], [], consStepSize, 0.3)
# polyak
# ii1, xx1, yy1, ff1 = miniBatchSGD([3, 3], 50, 2, data, [], [], [], [], polyak, 0.001)
# ii2, xx2, yy2, ff2 = miniBatchSGD([3, 3], 50, 5, data, [], [], [], [], polyak, 0.001)
# ii3, xx3, yy3, ff3 = miniBatchSGD([3, 3], 50, 10, data, [], [], [], [], polyak, 0.001)

# RMSProp
# ii1, xx1, yy1, ff1 = miniBatchSGD([3, 3], 50, 5, data, [], [], [], [], RMSProp, 0.01,0.25,0.001)
# ii2, xx2, yy2, ff2 = miniBatchSGD([3, 3], 50, 5, data, [], [], [], [], RMSProp, 0.05,0.25,0.001)
# ii3, xx3, yy3, ff3 = miniBatchSGD([3, 3], 50, 5, data, [], [], [], [], RMSProp, 0.05,0.99,0.001)
# ii4, xx4, yy4, ff4 = miniBatchSGD([3, 3], 50, 5, data, [], [], [], [], RMSProp, 0.2,0.99,0.001)
# ii5, xx5, yy5, ff5 = miniBatchSGD([3, 3], 50, 5, data, [], [], [], [], RMSProp, 0.34,0.99,0.001)
# ii6, xx6, yy6, ff6 = miniBatchSGD([3, 3], 50, 5, data, [], [], [], [], RMSProp, 0.34,0.999,0.001)
# ii1, xx1, yy1, ff1 = miniBatchSGD([3, 3], 50, 2, data, [], [], [], [], RMSProp, 0.34,0.99,0.001)
# ii2, xx2, yy2, ff2 = miniBatchSGD([3, 3], 50, 5, data, [], [], [], [], RMSProp, 0.34,0.99,0.001)
# ii3, xx3, yy3, ff3 = miniBatchSGD([3, 3], 50, 10, data, [], [], [], [], RMSProp, 0.34,0.99,0.001)

# heavyBall
# ii1, xx1, yy1, ff1 = miniBatchSGD([3, 3], 50, 5, data, [], [], [], [], heavyBall, 0.01, 0.25)
# ii2, xx2, yy2, ff2 = miniBatchSGD([3, 3], 50, 5, data, [], [], [], [], heavyBall, 0.1, 0.25)
# ii3, xx3, yy3, ff3 = miniBatchSGD([3, 3], 50, 5, data, [], [], [], [], heavyBall, 0.1, 0.5)
# ii4, xx4, yy4, ff4 = miniBatchSGD([3, 3], 50, 5, data, [], [], [], [], heavyBall, 0.3, 0.5)
# ii5, xx5, yy5, ff5 = miniBatchSGD([3, 3], 50, 5, data, [], [], [], [], heavyBall, 0.3, 0.9)
#
# ii1, xx1, yy1, ff1 = miniBatchSGD([3, 3], 50, 2, data, [], [], [], [], heavyBall, 0.1, 0.5)
# ii2, xx2, yy2, ff2 = miniBatchSGD([3, 3], 50, 5, data, [], [], [], [], heavyBall, 0.1, 0.5)
# ii3, xx3, yy3, ff3 = miniBatchSGD([3, 3], 50, 10, data, [], [], [], [], heavyBall, 0.1, 0.5)

# adam
# ii0, xx0, yy0, ff0 = miniBatchSGD([3, 3], 50, 5, data, [], [], [], [], adam, 0.1, 0.25,0.25,0.0001)
# ii1, xx1, yy1, ff1 = miniBatchSGD([3, 3], 50, 5, data, [], [], [], [], adam, 0.5, 0.25,0.25,0.0001)
# ii2, xx2, yy2, ff2 = miniBatchSGD([3, 3], 50, 5, data, [], [], [], [], adam, 0.5, 0.9,0.25,0.0001)
# ii3, xx3, yy3, ff3 = miniBatchSGD([3, 3], 50, 5, data, [], [], [], [], adam, 0.5, 0.25,0.9,0.0001)
# ii4, xx4, yy4, ff4 = miniBatchSGD([3, 3], 50, 5, data, [], [], [], [], adam, 0.5, 0.9,0.9,0.0001)
# ii5, xx5, yy5, ff5 = miniBatchSGD([3, 3], 50, 5, data, [], [], [], [], adam, 2, 0.9,0.9,0.0001)
# ii6, xx6, yy6, ff6 = miniBatchSGD([3, 3], 50, 5, data, [], [], [], [], adam, 5, 0.9,0.9,0.0001)

ii1, xx1, yy1, ff1 = miniBatchSGD([3, 3], 50, 2, data, [], [], [], [], adam, 2, 0.9,0.9,0.0001)
ii2, xx2, yy2, ff2 = miniBatchSGD([3, 3], 50, 5, data, [], [], [], [], adam, 2, 0.9,0.9,0.0001)
ii3, xx3, yy3, ff3 = miniBatchSGD([3, 3], 50, 10, data, [], [], [], [], adam, 2, 0.9,0.9,0.0001)

x = np.arange(-12, 5, 0.5)
y = np.arange(-16, 5, 0.5)

plt.figure(figsize=(7, 7))
# polyak
# plt.plot(ii, ff, color='b', label='constant, step size = 0.3,batch size = 5')
# plt.plot(ii2, ff2, color='g', label='polyak ,batch size = 5')
# plt.plot(ii3, ff3, color='c', label='polyak ,batch size = 10')
# plt.plot(xx, yy, color='b', label='constant, step size = 0.3,batch size = 5')
# plt.plot(xx1, yy1, color='r', label='polyak ,batch size = 2')
# plt.plot(xx2, yy2, color='g', label='polyak ,batch size = 5')
# plt.plot(xx3, yy3, color='c', label='polyak ,batch size = 10')

# RMSProp
# plt.plot(ii1, ff1, color='b', label='RMSProp,batch size = 5 alpha = 0.01 beta=0.25')
# plt.plot(ii2, ff2, color='r', label='RMSProp,batch size = 5 alpha = 0.05 beta=0.25')
# plt.plot(ii3, ff3, color='g', label='RMSProp,batch size = 5 alpha = 0.05 beta=0.99')
# plt.plot(ii4, ff4, color='c', label='RMSProp,batch size = 5 alpha = 0.2 beta=0.99')
# plt.plot(ii5, ff5, color='m', label='RMSProp,batch size = 5 alpha = 0.34 beta=0.99')
# plt.plot(ii6, ff6, color='y', label='RMSProp,batch size = 5 alpha = 0.5 beta=0.99')

# plt.plot(ii, ff, color='b', label='constant, step size = 0.3,batch size = 5')
# plt.plot(ii1, ff1, color='r', label='RMSProp,batch size = 2 alpha = 0.34 beta=0.99')
# plt.plot(ii2, ff2, color='g', label='RMSProp,batch size = 5 alpha = 0.34 beta=0.99')
# plt.plot(ii3, ff3, color='c', label='RMSProp,batch size = 10 alpha = 0.34 beta=0.99')

# plt.plot(xx1, yy1, color='b', label='RMSProp,batch size = 5 alpha = 0.01 beta=0.25')
# plt.plot(xx2, yy2, color='r', label='RMSProp,batch size = 5 alpha = 0.05 beta=0.25')
# plt.plot(xx3, yy3, color='g', label='RMSProp,batch size = 5 alpha = 0.05 beta=0.99')
# plt.plot(xx4, yy4, color='c', label='RMSProp,batch size = 5 alpha = 0.2 beta=0.99')
# plt.plot(xx5, yy5, color='m', label='RMSProp,batch size = 5 alpha = 0.34 beta=0.99')
# plt.plot(xx6, yy6, color='y', label='RMSProp,batch size = 5 alpha = 0.5 beta=0.99')

# plt.plot(xx, yy, color='b', label='constant, step size = 0.3,batch size = 5')
# plt.plot(xx1, yy1, color='r', label='RMSProp,batch size = 2 alpha = 0.34 beta=0.99')
# plt.plot(xx2, yy2, color='g', label='RMSProp,batch size = 5 alpha = 0.34 beta=0.99')
# plt.plot(xx3, yy3, color='c', label='RMSProp,batch size = 10 alpha = 0.34 beta=0.99')

# heavyBall
# plt.plot(ii1, ff1, color='b', label='heavyBall,batch size = 5 alpha = 0.01 beta=0.25')
# plt.plot(ii2, ff2, color='r', label='heavyBall,batch size = 5 alpha = 0.1 beta=0.25')
# plt.plot(ii3, ff3, color='g', label='heavyBall,batch size = 5 alpha = 0.1 beta=0.5')
# plt.plot(ii4, ff4, color='c', label='heavyBall,batch size = 5 alpha = 0.3 beta=0.5')
# plt.plot(ii5, ff5, color='m', label='heavyBall,batch size = 5 alpha = 0.3 beta=0.9')

# plt.plot(xx1, yy1, color='b', label='heavyBall,batch size = 5 alpha = 0.01 beta=0.25')
# plt.plot(xx2, yy2, color='r', label='heavyBall,batch size = 5 alpha = 0.1 beta=0.25')
# plt.plot(xx3, yy3, color='g', label='heavyBall,batch size = 5 alpha = 0.1 beta=0.5')
# plt.plot(xx4, yy4, color='c', label='heavyBall,batch size = 5 alpha = 0.3 beta=0.5')
# plt.plot(xx5, yy5, color='m', label='heavyBall,batch size = 5 alpha = 0.3 beta=0.9')

# plt.plot(ii, ff, color='b', label='constant, step size = 0.3,batch size = 5')
# plt.plot(ii1, ff1, color='r', label='heavyBall,batch size = 2 alpha = 0.1 beta=0.5')
# plt.plot(ii2, ff2, color='g', label='heavyBall,batch size = 5 alpha = 0.1 beta=0.5')
# plt.plot(ii3, ff3, color='c', label='heavyBall,batch size = 10 alpha = 0.1 beta=0.5')

# plt.plot(xx, yy, color='b', label='constant, step size = 0.3,batch size = 5')
# plt.plot(xx1, yy1, color='r', label='heavyBall,batch size = 2 alpha = 0.1 beta=0.5')
# plt.plot(xx2, yy2, color='g', label='heavyBall,batch size = 5 alpha = 0.1 beta=0.5')
# plt.plot(xx3, yy3, color='c', label='heavyBall,batch size = 10 alpha = 0.1 beta=0.5')

# adam
# plt.plot(ii0, ff0, color='black', label='adam,batch size = 5 alpha = 0.1 beta1=0.25 beta2=0.25')
# plt.plot(ii1, ff1, color='b', label='adam,batch size = 5 alpha = 0.5 beta1=0.25 beta2=0.25')
# plt.plot(ii2, ff2, color='r', label='adam,batch size = 5 alpha = 0.5 beta1=0.9 beta2=0.25')
# plt.plot(ii3, ff3, color='g', label='adam,batch size = 5 alpha = 0.5 beta1=0.25 beta2=0.9')
# plt.plot(ii4, ff4, color='c', label='adam,batch size = 5 alpha = 0.5 beta1=0.9 beta2=0.9')
# plt.plot(ii5, ff5, color='m', label='adam,batch size = 5 alpha = 2 beta1=0.9 beta2=0.9')
# plt.plot(ii6, ff6, color='y', label='adam,batch size = 5 alpha = 5 beta1=0.9 beta2=0.9')

# plt.plot(ii, ff, color='b', label='constant, step size = 0.3,batch size = 5')
# plt.plot(ii1, ff1, color='r', label='adam,batch size = 2 alpha = 2 beta1=0.9 beta2=0.9')
# plt.plot(ii2, ff2, color='g', label='adam,batch size = 5 alpha = 2 beta1=0.9 beta2=0.9')
# plt.plot(ii3, ff3, color='c', label='adam,batch size = 10 alpha = 2 beta1=0.9 beta2=0.9')

plt.plot(xx, yy, color='b', label='constant, step size = 0.3,batch size = 5')
plt.plot(xx1, yy1, color='r', label='adam,batch size = 2 alpha = 2 beta1=0.9 beta2=0.9')
plt.plot(xx2, yy2, color='g', label='adam,batch size = 5 alpha = 2 beta1=0.9 beta2=0.9')
plt.plot(xx3, yy3, color='c', label='adam,batch size = 10 alpha = 2 beta1=0.9 beta2=0.9')


X, Y = np.meshgrid(x, y)
Z = np.zeros((len(X), len(X[0])))
for i, v in enumerate(X):
    for j, v2 in enumerate(v):
        Z[i][j] = f((X[i][j], Y[i][j]), data)

contours = plt.contour(X, Y, Z, 20);
plt.clabel(contours, inline=True, fontsize=10)
# plt.xlabel('epoches')
# plt.ylabel('function value of log')
plt.xlabel('x1')
plt.ylabel('x2')
# plt.yscale("log")
# plt.xscale("log")
# plt.ylim((-0.5, 0.1))
plt.legend()
plt.show()
