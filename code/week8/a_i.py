import random
import time

import numpy as np

from minheap import BtmkHeap
import sympy
from matplotlib import pyplot as plt
from sympy import Max

x1, y1 = sympy.symbols('x,y', real=True)
xa = sympy.Array([x1, y1])
f1 = 1 * (x1 - 5) ** 4 + 3 * (y1 - 0) ** 2
f2 = Max(x1 - 5, 0) + 3 * abs(y1 - 0)

dfdx1 = sympy.diff(f1, xa)
dfdx1 = sympy.lambdify(xa, dfdx1)

f1 = sympy.lambdify(xa, f1)
f2 = sympy.lambdify(xa, f2)

x2 = sympy.symbols('x', real=True)
f2p2 = 3 * abs(x2 - 0)
dfdx2 = sympy.diff(f2p2, x2)
dfdx2 = sympy.lambdify(x2, dfdx2)


def getF1(x, y):
    return f1(x, y)


def getF2(x, y):
    return f2(x, y)


def getDff2p1(x):
    if x > 5:
        return 1
    else:
        return 0


def getDff2(x, y):
    dx = getDff2p1(x)
    dy = dfdx2(y)
    return dx, dy

def getPF1(x, y):
    x1 = x.copy()
    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[1]):
            if x[i][j] <= 5:
                x1[i][j] = 0
            else:
                x1[i][j] = x[i][j] - 5


    part2 = 3 * abs(y)

    return x1 + part2


def gradientDescent(x, y, alpha, iters, ii, ss, tt, xx, yy, ff):
    start_time = time.time() * 1000
    i = 0
    s = 0
    while i < iters:
        # dx, dy = dfdx1(x, y)
        dx, dy = getDff2(x, y)
        stepx = alpha * dx
        stepy = alpha * dy
        ii.append(i)
        s = s + 2
        ss.append(s)
        xx.append(x)
        yy.append(y)
        value = f2(x, y)
        ff.append(value)
        used = (time.time() * 1000 - start_time)
        print(used)
        tt.append(used)
        x = x - stepx
        y = y - stepy
        i = i + 1
    return ii, ss, tt, xx, yy, ff


def globalRandomSearch(x, y, iters, ii, ss, tt, xx, yy, ff):
    start_time = time.time() * 1000
    minValue = float('inf')
    minX = random.uniform(x[0], x[1])
    minY = random.uniform(y[0], y[1])
    i = 0
    s = 0
    while i < iters:
        cx = random.uniform(x[0], x[1])
        cy = random.uniform(y[0], y[1])
        value = f2(cx, cy)
        if minValue > value:
            minValue = value
            minX = cx
            minY = cy
        ii.append(i)
        s = s + 1
        ss.append(s)
        xx.append(minX)
        yy.append(minY)
        ff.append(minValue)
        used = (time.time() * 1000 - start_time)
        # print(used)
        tt.append(used)
        i = i + 1
    return ii, ss, tt, xx, yy, ff


def populationBasedSearch(x, y, iters, Nsample, bestP, range, exploitP, ii, ss, tt, xx, yy, ff):
    # choose Nsample points
    start_time = time.time() * 1000
    stp = BtmkHeap(bestP)
    i = 0
    s = 0
    while i < Nsample:
        cx = random.uniform(x[0], x[1])
        cy = random.uniform(y[0], y[1])
        value = f2(cx, cy)
        stp.Push((value, [cx, cy]))
        i = i + 1

    i = 0
    while i < iters:
        datas = stp.BtmK()
        ii.append(i)
        used = (time.time() * 1000 - start_time)
        # print(used)
        tt.append(used)
        smallestXY = stp.getSmallestV()
        xx.append(smallestXY[0])
        yy.append(smallestXY[1])
        ff.append(stp.getSmallest())
        for data in datas:
            xy = data[1]
            j = 0
            while j < exploitP:
                cx = random.uniform(xy[0] - range, xy[0] + range)
                cy = random.uniform(xy[1] - range, xy[1] + range)
                value = f2(cx, cy)
                stp.Push((value, [cx, cy]))
                j = j + 1
        i = i + 1

    return ii, ss, tt, xx, yy, ff


ii, ss, tt, xx, yy, ff = gradientDescent(2, 2, 0.03, 1000, [], [], [], [], [], [])
# ii1, ss1, tt1, xx1, yy1, ff1 = gradientDescent(2, 2, 0.01, 1000, [], [], [], [], [], [])
ii2, ss2, tt2, xx2, yy2, ff2 = globalRandomSearch([4, 6], [-1, 1], 1000, [], [], [], [], [], [])
# ii3, ss3, tt3, xx3, yy3, ff3 = globalRandomSearch([3, 7], [-2, 2], 15000, [], [], [], [], [], [])
# ii4, ss4, tt4, xx4, yy4, ff4 = globalRandomSearch([0, 10], [-5, 5], 15000, [], [], [], [], [], [])
ii3, ss3, tt3, xx3, yy3, ff3 = populationBasedSearch([4, 6], [-1, 1], 1000, 50, 5, 0.1, 5, [], [], [], [], [], [])
ii4, ss4, tt4, xx4, yy4, ff4 = populationBasedSearch([3, 7], [-2, 2], 1000, 50, 5, 0.1, 5, [], [], [], [], [], [])
ii5, ss5, tt5, xx5, yy5, ff5 = populationBasedSearch([0, 10], [-5, 5], 1000, 50, 5, 0.1, 5, [], [], [], [], [], [])


plt.figure(figsize=(7, 7))

x = np.arange(1, 7, 0.5)
y = np.arange(-1, 3, 0.5)
X, Y = np.meshgrid(x, y)

contours = plt.contour(X, Y, getPF1(X, Y), 5);
plt.clabel(contours, inline=True, fontsize=10)
# plt.plot(tt, ff, color='b', label='gradient Descent alpha = 0.03')
# plt.plot(tt1, ff1, color='r', label='gradient Descent alpha = 0.01')
# plt.plot(tt2, ff2, color='g', label='global random search x=[4,6] y=[-1,1]')
# plt.plot(tt3, ff3, color='y', label='global random search x=[3,7] y=[-2,2]')
# plt.plot(tt4, ff4, color='c', label='global random search x=[0,10] y=[-5,5]')
# plt.plot(tt1, ss1, color='r', label='gradient Descent alpha = 0.01')
# plt.plot(tt2, ss2, color='g', label='global random search x=[4,6] y=[-1,1]')
# plt.plot(tt, ff, color='b', label='gradient Descent alpha = 0.03 iter=1000')
# plt.plot(tt2, ff2, color='r', label='global random search x=[4,6] y=[-1,1] iter=1000')
# plt.plot(tt3, ff3, color='g', label='population based search x=[4,6] y=[-1,1] iter=1000')
# plt.plot(tt4, ff4, color='c', label='population based search x=[3,7] y=[-2,2] iter=1000')
# plt.plot(tt5, ff5, color='m', label='population based search x=[0,10] y=[-5,5] iter=1000')
plt.plot(xx, yy, color='b', label='gradient Descent alpha = 0.03 iter=1000')
plt.plot(xx2, yy2, color='r', label='global random search x=[4,6] y=[-1,1] iter=1000')
plt.plot(xx3, yy3, color='g', label='population based search x=[4,6] y=[-1,1] iter=1000')
plt.plot(xx4, yy4, color='c', label='population based search x=[3,7] y=[-2,2] iter=1000')
plt.plot(xx5, yy5, color='m', label='population based search x=[0,10] y=[-5,5] iter=1000')
plt.xlabel('x')
plt.ylabel('y')
# plt.xlabel('runing time(millisecond)')
# plt.ylabel('number of Derivative function/function computations')
# plt.xlim((0, 30))
# plt.yscale("log")
plt.legend()
plt.show()
