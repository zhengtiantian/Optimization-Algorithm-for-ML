from typing import overload

import numpy as np
import sympy
import math

from matplotlib import pyplot as plt
from sympy import Max

x = sympy.symbols('x', real=True)
f = x ** 2
dfdx = sympy.diff(f, x)
f = sympy.lambdify(x, f)
dfdx = sympy.lambdify(x, dfdx)

x1, y1 = sympy.symbols('x,y', real=True)
xa = sympy.Array([x1, y1])
# f1 = 1 * (x1 - 5) ** 4 + 3 * (y1 - 0) ** 2
# f1 = (1 - x1) ** 2 + 100 * (y1 - x1 ** 2) ** 2
# f1 = abs(x1) + y1 ** 2
# f1 = 0.5 * (x1 ** 2 + 10 * y1 ** 2)
f1 = Max(x1 - 5, 0) + 3 * abs(y1 - 0)
dfdx1 = sympy.diff(f1, xa)
# dfdx2 = sympy.diff(f2, xa)
print(dfdx1)

f1 = sympy.lambdify(xa, f1)
print(f1(1, 1))
# f2 = sympy.lambdify(xa, f2)
dfdx1 = sympy.lambdify(xa, dfdx1)
# dfdx2 = sympy.lambdify(xa, dfdx2)


x2 = sympy.symbols('x', real=True)
f2 = 3 * abs(x2 - 0)
dfdx2 = sympy.diff(f2, x2)
dfdx2 = sympy.lambdify(x2, dfdx2)
print(dfdx2(2))


# print(dfdx2)

# f = sympy.lambdify(x, f)
# dfdx = sympy.lambdify(x, dfdx)


def getF1(x):
    return f(x)


def getF2(x, y):
    return f1(x, y)


def getDf1(x):
    return dfdx(x)


# def getDf2(x, y):
#     return dfdx1(x, y)
def getDf(x):
    if x > 5:
        return 1
    else:
        return 0


def getDf2(x, y):
    dx = getDf(x)
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


def polyak(startX, startY, maxIters, epsilon, ii, ss):
    x = startX
    i = 0
    if startY is not None:
        y = startY
        while i < maxIters:
            numerator = getF2(x, y) - 0
            dx, dy = getDf2(x, y)
            alpha = numerator / (dx ** 2 + dy ** 2 + epsilon)
            x = x - alpha * dx
            y = y - alpha * dy
            i = i + 1
            ii.append(i)
            ss.append(getF2(x, y))
    else:
        while i < maxIters:
            numerator = getF1(x) - 0
            alpha = numerator / (getDf1(x) ** 2 + epsilon)
            x = x - alpha * getDf1(x)
            i = i + 1
            ii.append(i)
            ss.append(alpha)
    return ii, ss


def RMSProp(startX, startY, maxIters, alpha0, Beta, epsilon, ii, ff, ssx, ssy, xx, yy):
    x = startX
    i = 0
    sum = 0
    if startY is None:
        while i < maxIters:
            sum = Beta * sum + (1 - Beta) * getDf1(x) ** 2
            alpha = alpha0 / (math.sqrt(sum) + epsilon)
            x = x - alpha * getDf1(x)
            i = i + 1
            ii.append(i)
            ff.append(alpha)
    else:
        y = startY
        sum2 = 0
        while i < maxIters:

            dx, dy = getDf2(x, y)
            sum = Beta * sum + (1 - Beta) * (dx ** 2)
            sum2 = Beta * sum2 + (1 - Beta) * (dy ** 2)
            alpha = alpha0 / (math.sqrt(sum) + epsilon)
            alpha2 = alpha0 / (math.sqrt(sum2) + epsilon)
            x = x - alpha * dx
            y = y - alpha2 * dy
            i = i + 1
            ii.append(i)
            ff.append(getF2(x, y))
            ssx.append(alpha)
            ssy.append(alpha2)
            xx.append(x)
            yy.append(y)
    return ii, ff, ssx, ssy, xx, yy


def heavyBall(startX, startY, maxIters, alpha, Beta, ii, ff, ssx, ssy, xx, yy):
    x = startX
    i = 0
    z = 0
    if startY is None:
        while i < maxIters:
            z = Beta * z + alpha * getDf1(x)
            x = x - z
            i = i + 1
            ii.append(i)
            ff.append(getF1(x))
    else:
        y = startY
        z2 = 0
        while i < maxIters:
            dx, dy = getDf2(x, y)
            z = Beta * z + alpha * dx
            z2 = Beta * z2 + alpha * dy
            x = x - z
            y = y - z2
            i = i + 1
            ii.append(i)
            ff.append(getF2(x, y))
            ssx.append(z)
            ssy.append(z2)
            xx.append(x)
            yy.append(y)
    return ii, ff, ssx, ssy, xx, yy


def Adam(startX, startY, maxIters, alpha, Beta1, Beta2, epsilon, ii, ff, ssx, ssy, xx, yy):
    x = startX
    i = 0
    mt = 0
    vt = 0
    if startY is None:
        while i < maxIters:
            mt = Beta1 * mt + (1 - Beta1) * getDf1(x)
            vt = Beta2 * vt + (1 - Beta2) * (getDf1(x) ** 2)
            me = mt / (1 - Beta1 ** (i + 1))
            ve = vt / (1 - Beta2 ** (i + 1))
            x = x - alpha * (me / (math.sqrt(ve) + epsilon))
            i = i + 1
            ii.append(i)
            ff.append(getF1(x))

    else:
        y = startY
        mt2 = 0
        vt2 = 0
        while i < maxIters:
            dx, dy = getDf2(x, y)
            mt = Beta1 * mt + (1 - Beta1) * dx
            mt2 = Beta1 * mt2 + (1 - Beta1) * dy
            vt = Beta2 * vt + (1 - Beta2) * (dx ** 2)
            vt2 = Beta2 * vt2 + (1 - Beta2) * (dy ** 2)
            me = mt / (1 - Beta1 ** (i + 1))
            me2 = mt2 / (1 - Beta1 ** (i + 1))
            ve = vt / (1 - Beta2 ** (i + 1))
            ve2 = vt2 / (1 - Beta2 ** (i + 1))
            x = x - alpha * (me / (math.sqrt(ve) + epsilon))
            y = y - alpha * (me2 / (math.sqrt(ve2) + epsilon))
            i = i + 1
            ii.append(i)
            ff.append(getF2(x, y))
            ssx.append(alpha * (me / (math.sqrt(ve) + epsilon)))
            ssy.append(alpha * (me2 / (math.sqrt(ve2) + epsilon)))
            xx.append(x)
            yy.append(y)
    return ii, ff, ssx, ssy, xx, yy

plt.figure(figsize=(7, 7))
# polyak(1, None, 50, 0.0001)
# polyak(-1.25,0.5, 1500, 0.0001)

# RMSProp(1,None,50,0.06,0.9,0.0001)
# ii, ff, ssx, ssy, xx, yy = RMSProp(1, 1, 500, 0.01, 0.25, 0.0001, [], [], [], [], [], [])
# ii1, ff1, ssx1, ssy1, xx1, yy1 = RMSProp(1, 1, 500, 0.2, 0.25, 0.0001, [], [], [], [], [], [])
# ii2, ff2, ssx2, ssy2, xx2, yy2 = RMSProp(1, 1, 500, 0.2, 0.5, 0.0001, [], [], [], [], [], [])
# ii3, ff3, ssx3, ssy3, xx3, yy3 = RMSProp(1, 1, 500, 0.5, 0.5, 0.0001, [], [], [], [], [], [])
# ii4, ff4, ssx4, ssy4, xx4, yy4 = RMSProp(1, 1, 500, 0.5, 0.9, 0.0001, [], [], [], [], [], [])
# ii, ff, ssx, ssy, xx, yy = RMSProp(1, 1, 100, 0.1, 0.25, 0.0001, [], [], [], [], [], [])
# ii1, ff1, ssx1, ssy1, xx1, yy1 = RMSProp(1, 1, 100, 0.2, 0.25, 0.0001, [], [], [], [], [], [])
# ii2, ff2, ssx2, ssy2, xx2, yy2 = RMSProp(1, 1, 100, 0.3, 0.25, 0.0001, [], [], [], [], [], [])
# ii3, ff3, ssx3, ssy3, xx3, yy3 = RMSProp(1, 1, 100, 0.1, 0.9, 0.0001, [], [], [], [], [], [])
# ii4, ff4, ssx4, ssy4, xx4, yy4 = RMSProp(1, 1, 80, 0.2, 0.9, 0.0001, [], [], [], [], [], [])
# ii5, ff5, ssx5, ssy5, xx5, yy5 = RMSProp(1, 1, 80, 0.3, 0.9, 0.0001, [], [], [], [], [], [])
# ii5, ff5, ssx5, ssy5, xx5, yy5 = RMSProp(1, 1, 200, 0.9, 0.9, 0.0001, [], [], [], [], [], [])
# RMSProp(0.02, 0.1, 1000, 0.01, 0.9, 0.00001)
# heavyBall(1, None, 50, 1, 0.25)
# ii, ff, ssx, ssy, xx, yy = heavyBall(1, 1, 200, 0.001, 0.25, [], [], [], [], [], [])
# ii1, ff1, ssx1, ssy1, xx1, yy1 = heavyBall(1, 1, 200, 0.005, 0.25, [], [], [], [], [], [])
# ii2, ff2, ssx2, ssy2, xx2, yy2 = heavyBall(1, 1, 200, 0.005, 0.5, [], [], [], [], [], [])
# ii3, ff3, ssx3, ssy3, xx3, yy3 = heavyBall(1, 1, 200, 0.01, 0.5, [], [], [], [], [], [])
# ii4, ff4, ssx4, ssy4, xx4, yy4 = heavyBall(1, 1, 200, 0.01, 0.9, [], [], [], [], [], [])
# ii, ff, ssx, ssy, xx, yy = heavyBall(1, 1, 100, 1, 0.25, [], [], [], [], [], [])
# ii1, ff1, ssx1, ssy1, xx1, yy1 = heavyBall(1, 1, 500, 0.005, 0.25, [], [], [], [], [], [])
# ii2, ff2, ssx2, ssy2, xx2, yy2 = heavyBall(1, 1, 500, 0.01, 0.5, [], [], [], [], [], [])
# ii3, ff3, ssx3, ssy3, xx3, yy3 = heavyBall(1, 1, 500, 0.001, 0.9, [], [], [], [], [], [])
# ii4, ff4, ssx4, ssy4, xx4, yy4 = heavyBall(1, 1, 500, 0.005, 0.9, [], [], [], [], [], [])
# ii5, ff5, ssx5, ssy5, xx5, yy5 = heavyBall(1, 1, 500, 0.01, 0.9, [], [], [], [], [], [])
# ii, ff, ssx, ssy, xx, yy = heavyBall(1, 1, 5, 0.05, 0.9, [], [], [], [], [], [])
# Adam(1.0,None, 50, 0.1, 0.9, 0.999, 0)
# ii, ff, ssx, ssy, xx, yy = Adam(1, 1, 500, 10, 0.9, 0.9, 0.0001, [], [], [], [], [], [])
# ii1, ff1, ssx1, ssy1, xx1, yy1 = Adam(1, 1, 150, 0.2, 0.25, 0.25, 0.0001, [], [], [], [], [], [])
# ii2, ff2, ssx2, ssy2, xx2, yy2 = Adam(1, 1, 150, 0.2, 0.5, 0.25, 0.0001, [], [], [], [], [], [])
# ii3, ff3, ssx3, ssy3, xx3, yy3 = Adam(1, 1, 150, 0.2, 0.5, 0.5, 0.0001, [], [], [], [], [], [])
# ii4, ff4, ssx4, ssy4, xx4, yy4 = Adam(1, 1, 150, 0.9, 0.9, 0.9, 0.0001, [], [], [], [], [], [])
# ii, ff, ssx, ssy, xx, yy = Adam(1, 1, 200, 0.1, 0.25, 0.25, 0.0001, [], [], [], [], [], [])
# ii1, ff1, ssx1, ssy1, xx1, yy1 = Adam(1, 1, 200, 0.5, 0.25, 0.25, 0.0001, [], [], [], [], [], [])
# ii2, ff2, ssx2, ssy2, xx2, yy2 = Adam(1, 1, 200, 0.9, 0.25, 0.25, 0.0001, [], [], [], [], [], [])
# ii3, ff3, ssx3, ssy3, xx3, yy3 = Adam(1, 1, 200, 0.1, 0.9, 0.9, 0.0001, [], [], [], [], [], [])
# ii4, ff4, ssx4, ssy4, xx4, yy4 = Adam(1, 1, 200, 0.5, 0.9, 0.9, 0.0001, [], [], [], [], [], [])
# ii5, ff5, ssx5, ssy5, xx5, yy5 = Adam(1, 1, 200, 0.9, 0.9, 0.9, 0.0001, [], [], [], [], [], [])
# ii, ff, ssx, ssy, xx, yy = Adam(1, 1, 2000, 0.3, 1.1, 0.95, 0.0001, [], [], [], [], [], [])


# plt.plot(ii, ff, color='b', label='function value with alpha=0.01, beta = 0.25')
# plt.plot(ii1, ff1, color='r', label='function value with alpha=0.2, beta = 0.25')
# plt.plot(ii2, ff2, color='g', label='function value with alpha=0.2, beta = 0.5')
# plt.plot(ii3, ff3, color='y', label='function value with alpha=0.5, beta = 0.5')
# plt.plot(ii4, ff4, color='c', label='function value with alpha=0.5, beta = 0.9')
# plt.plot(ii, ff, color='b', label='function value with alpha=0.1, beta = 0.25')
# plt.plot(ii1, ff1, color='r', label='function value with alpha=0.2, beta = 0.25')
# plt.plot(ii2, ff2, color='g', label='function value with alpha=0.3, beta = 0.25')
# plt.plot(ii3, ff3, color='y', label='function value with alpha=0.1, beta = 0.9')
# plt.plot(ii4, ff4, color='c', label='function value with alpha=0.2, beta = 0.9')
# plt.plot(ii5, ff5, color='m', label='function value with alpha=0.3, beta = 0.9')
# plt.plot(ii5, ff5, color='c', label='function value with alpha=5000, beta=0.9, epsilon=0.0001')
# plt.plot(ii, ssx, color='b', label='x step size with alpha=0.06, beta = 0.25')
# plt.plot(ii, ssy, color='r', label='y step size with alpha=0.06, beta = 0.25')
# plt.plot(ii1, ssx1, color='g', label='x step size with alpha=0.2, beta = 0.25')
# plt.plot(ii1, ssy1, color='y', label='y step size with alpha=0.2, beta = 0.25')
# plt.plot(ii2, ssx2, color='c', label='x step size with alpha=0.2, beta = 0.5')
# plt.plot(ii2, ssy2, color='m', label='y step size with alpha=0.2, beta = 0.5')
# x = np.arange(1, 6, 0.1)
# y = np.arange(-2, 2, 0.1)
# X, Y = np.meshgrid(x, y)
# contours = plt.contour(X, Y, getF2(X, Y), 20);
# plt.clabel(contours, inline=True, fontsize=10)
# plt.plot(xx,yy,color='b', label='the path of x and y with alpha=0.1, beta = 0.25')
# plt.plot(xx2,yy2,color='r', label='the path of x and y with alpha=0.9, beta = 0.25')
# plt.plot(xx3,yy3,color='g', label='the path of x and y with alpha=0.1, beta = 0.9')
# plt.plot(xx5,yy5,color='y', label='the path of x and y with alpha=0.9, beta = 0.9')

# plt.plot(ii, ff, color='b', label='function value with alpha=0.001, beta = 0.25')
# plt.plot(ii1, ff1, color='r', label='function value with alpha=0.005, beta = 0.25')
# plt.plot(ii2, ff2, color='g', label='function value with alpha=0.005, beta = 0.5')
# plt.plot(ii3, ff3, color='y', label='function value with alpha=0.01, beta = 0.5')
# plt.plot(ii4, ff4, color='c', label='function value with alpha=0.01, beta = 0.9')
# plt.plot(ii, ff, color='b', label='function value with alpha=0.001, beta = 0.25')
# plt.plot(ii1, ff1, color='r', label='function value with alpha=0.005, beta = 0.25')
# plt.plot(ii2, ff2, color='g', label='function value with alpha=0.008, beta = 0.25')
# plt.plot(ii3, ff3, color='y', label='function value with alpha=0.001, beta = 0.9')
# plt.plot(ii4, ff4, color='c', label='function value with alpha=0.005, beta = 0.9')
# plt.plot(ii5, ff5, color='m', label='function value with alpha=0.01, beta = 0.9')
# plt.plot(ii, ssy, color='b', label='y step with alpha=0.001, beta = 0.25')
# plt.plot(ii1, ssy1, color='r', label='y step with alpha=0.005, beta = 0.25')
# plt.plot(ii2, ssy2, color='g', label='y step with alpha=0.005, beta = 0.5')
# plt.plot(ii, ssy3, color='y', label='y step with alpha=0.01, beta = 0.5')
# plt.plot(ii, ssy4, color='c', label='y step with alpha=0.01, beta = 0.9')

# plt.plot(xx,yy,color='b', label='the path of x and y with alpha=0.001, beta = 0.25')
# plt.plot(xx1,yy1,color='r', label='the path of x and y with alpha=0.005, beta = 0.25')
# plt.plot(xx2,yy2,color='g', label='the path of x and y with alpha=0.005, beta = 0.5')
# plt.plot(xx3,yy3,color='y', label='the path of x and y with alpha=0.01, beta = 0.5')
# plt.plot(xx4,yy4,color='c', label='the path of x and y with alpha=0.01, beta = 0.9')
# plt.plot(ii, ff, color='b', label='function value with alpha=0.05, beta = 0.9')

# plt.plot(ii, ff, color='b', label='function value with alpha=10,beta1 = 0.9,beta2 = 0.9')
# plt.plot(ii1, ff1, color='r', label='function value with alpha=0.2,beta1 = 0.25,beta2 = 0.25')
# plt.plot(ii2, ff2, color='g', label='function value with alpha=0.2,beta1 = 0.5,beta2 = 0.25')
# plt.plot(ii3, ff3, color='y', label='function value with alpha=0.2,beta1 = 0.5,beta2 = 0.5')
# plt.plot(ii4, ff4, color='c', label='function value with alpha=0.3,beta1 = 0.9,beta2 = 0.9')
# plt.plot(ii, ff, color='b', label='function value with alpha=0.1,beta1 = 0.9,beta2 = 0.9')
# plt.plot(ii1, ff1, color='r', label='function value with alpha=0.5,beta1 = 0.25,beta2 = 0.25')
# plt.plot(ii2, ff2, color='g', label='function value with alpha=0.9,beta1 = 0.25,beta2 = 0.25')
# plt.plot(ii3, ff3, color='y', label='function value with alpha=0.1,beta1 = 0.9,beta2 = 0.9')
# plt.plot(ii4, ff4, color='c', label='function value with alpha=0.5,beta1 = 0.9,beta2 = 0.9')
# plt.plot(ii5, ff5, color='m', label='function value with alpha=0.9,beta1 = 0.9,beta2 = 0.9')
# plt.plot(ii, ssx, color='b', label='x step with alpha=0.1, beta = 0.25,beta2 = 0.25')
# plt.plot(ii, ssy, color='r', label='y step with alpha=0.1, beta = 0.25,beta2 = 0.25')
# plt.plot(ii1, ssy1, color='b', label='y step with alpha=0.2, beta = 0.25,beta2 = 0.25')
# plt.plot(ii2, ssy2, color='g', label='y step with alpha=0.2, beta = 0.5,beta2 = 0.25')
# plt.plot(ii3, ssy3, color='y', label='y step with alpha=0.2, beta = 0.5,beta2 = 0.5')
# plt.plot(xx, yy, color='b', label='the path of x and y with alpha=0.1, beta = 0.25,beta2 = 0.25')
# plt.plot(xx1, yy1, color='r', label='the path of x and y with alpha=0.2, beta = 0.25,beta2 = 0.25')
# plt.plot(xx2, yy2, color='g', label='the path of x and y with alpha=0.2, beta = 0.5,beta2 = 0.25')
# plt.plot(xx3, yy3, color='y', label='the path of x and y with alpha=0.2, beta = 0.5,beta2 = 0.5')
# plt.plot(xx4, yy4, color='c', label='the path of x and y with alpha=0.3, beta = 0.9,beta2 = 0.9')
# plt.plot(ii, ff, color='b', label='function value with alpha=0.3, beta1 = 1.1,beta2 = 0.95')

# Max(x1 - 5, 0) + 3 * abs(y1 - 0)
# ii, ff, ssx, ssy, xx, yy = RMSProp(10, 1, 200, 0.06, 0.25, 0.0001, [], [], [], [], [], [])
# ii1, ff1, ssx1, ssy1, xx1, yy1 = RMSProp(10, 1, 200, 0.2, 0.25, 0.0001, [], [], [], [], [], [])
# ii2, ff2, ssx2, ssy2, xx2, yy2 = RMSProp(10, 1, 200, 0.2, 0.5, 0.0001, [], [], [], [], [], [])
# ii3, ff3, ssx3, ssy3, xx3, yy3 = RMSProp(10, 1, 200, 0.5, 0.5, 0.0001, [], [], [], [], [], [])
# ii4, ff4, ssx4, ssy4, xx4, yy4 = RMSProp(10, 1, 200, 0.5, 0.9, 0.0001, [], [], [], [], [], [])
ii5, ff5, ssx5, ssy5, xx5, yy5 = RMSProp(10, 1, 50, 0.9, 0.25, 0.1, [], [], [], [], [], [])
# plt.plot(ii, ff, color='b', label='function value with alpha=0.06, beta = 0.25')
# plt.plot(ii1, ff1, color='r', label='function value with alpha=0.2, beta = 0.25')
# plt.plot(ii2, ff2, color='g', label='function value with alpha=0.2, beta = 0.5')
# plt.plot(ii3, ff3, color='y', label='function value with alpha=0.5, beta = 0.5')
# plt.plot(ii4, ff4, color='c', label='function value with alpha=0.5, beta = 0.9')
plt.plot(ii5, ff5, color='c', label='function value with alpha=0.2, beta=1.0, epsilon=0.1')
# plt.plot(ii, ssx, color='b', label='x step size with alpha=0.06, beta = 0.25')
# plt.plot(ii, ssy, color='r', label='y step size with alpha=0.06, beta = 0.25')
# plt.plot(ii1, ssx1, color='g', label='x step size with alpha=0.2, beta = 0.25')
# plt.plot(ii1, ssy1, color='y', label='y step size with alpha=0.2, beta = 0.25')
# plt.plot(ii2, ssx2, color='c', label='x step size with alpha=0.2, beta = 0.5')
# plt.plot(ii2, ssy2, color='m', label='y step size with alpha=0.2, beta = 0.5')
# plt.plot(ii3, ssy3, color='gold', label='x step size with alpha=0.2, beta = 0.5')
# plt.plot(ii3, ssy3, color='k', label='y step size with alpha=0.2, beta = 0.5')
# x = np.arange(4, 12, 0.1)
# y = np.arange(-1, 2, 0.1)
# X, Y = np.meshgrid(x, y)
# contours = plt.contour(X, Y, getPF1(X, Y), 20);
# plt.clabel(contours, inline=True, fontsize=10)
# plt.plot(xx, yy, color='b', label='the path of x and y with alpha=0.06, beta = 0.25')
# plt.plot(xx1, yy1, color='r', label='the path of x and y with alpha=0.2, beta = 0.25')
# plt.plot(xx2, yy2, color='g', label='the path of x and y with alpha=0.2, beta = 0.5')
# plt.plot(xx3, yy3, color='y', label='the path of x and y with alpha=0.5, beta = 0.5')

# ii, ff, ssx, ssy, xx, yy = heavyBall(10, 1, 500, 0.01, 0.25, [], [], [], [], [], [])
# ii1, ff1, ssx1, ssy1, xx1, yy1 = heavyBall(10, 1, 500, 0.05, 0.25, [], [], [], [], [], [])
# ii2, ff2, ssx2, ssy2, xx2, yy2 = heavyBall(10, 1, 500, 0.01, 0.9, [], [], [], [], [], [])
# ii3, ff3, ssx3, ssy3, xx3, yy3 = heavyBall(10, 1, 500, 0.05, 0.9, [], [], [], [], [], [])
# ii4, ff4, ssx4, ssy4, xx4, yy4 = heavyBall(10, 5, 500, 0.06, 0.9, [], [], [], [], [], [])
# ii, ff, ssx, ssy, xx, yy = heavyBall(10, 1, 100, 0.6, 0.9, [], [], [], [], [], [])
# plt.plot(ii, ff, color='b', label='function value with alpha=0.01, beta = 0.25')
# plt.plot(ii1, ff1, color='r', label='function value with alpha=0.05, beta = 0.25')
# plt.plot(ii2, ff2, color='g', label='function value with alpha=0.01, beta = 0.9')
# plt.plot(ii3, ff3, color='y', label='function value with alpha=0.05, beta = 0.9')
# plt.plot(ii4, ff4, color='c', label='function value with alpha=0.06, beta = 0.9')
# plt.plot(ii, ssy, color='b', label='y step with alpha=0.01, beta = 0.25')
# plt.plot(ii1, ssy1, color='r', label='y step with alpha=0.03, beta = 0.25')
# plt.plot(ii2, ssy2, color='g', label='y step with alpha=0.03, beta = 0.5')
# plt.plot(ii, ssy3, color='y', label='y step with alpha=0.06, beta = 0.5')
# plt.plot(ii, ssy4, color='c', label='y step with alpha=0.06, beta = 0.9')

# x = np.arange(-2, 11, 0.1)
# y = np.arange(-0.5, 1.5, 0.1)
# X, Y = np.meshgrid(x, y)
# contours = plt.contour(X, Y, getPF1(X, Y), 20);
# plt.clabel(contours, inline=True, fontsize=10)
# plt.plot(xx,yy,color='b', label='the path of x and y with alpha=0.01, beta = 0.25,500 iters')
# plt.plot(xx1,yy1,color='r', label='the path of x and y with alpha=0.03, beta = 0.25,500 iters')
# plt.plot(xx2,yy2,color='g', label='the path of x and y with alpha=0.03, beta = 0.5,500 iters')
# plt.plot(xx3,yy3,color='y', label='the path of x and y with alpha=0.06, beta = 0.5,500 iters')
# plt.plot(xx4,yy4,color='c', label='the path of x and y with alpha=0.06, beta = 0.9,500 iters')
# plt.plot(ii, ff, color='b', label='function value with alpha=0.06, beta = 1')

# ii, ff, ssx, ssy, xx, yy = Adam(10, 1, 500, 0.1, 0.25, 0.25, 0.0001, [], [], [], [], [], [])
# ii1, ff1, ssx1, ssy1, xx1, yy1 = Adam(10, 1, 500, 0.2, 0.25, 0.25, 0.0001, [], [], [], [], [], [])
# ii2, ff2, ssx2, ssy2, xx2, yy2 = Adam(10, 1, 500, 0.2, 0.5, 0.25, 0.0001, [], [], [], [], [], [])
# ii3, ff3, ssx3, ssy3, xx3, yy3 = Adam(10, 1, 500, 0.2, 0.5, 0.5, 0.0001, [], [], [], [], [], [])
# ii4, ff4, ssx4, ssy4, xx4, yy4 = Adam(10, 1, 500, 0.2, 0.9, 0.9, 0.0001, [], [], [], [], [], [])
# ii5, ff5, ssx5, ssy5, xx5, yy5 = Adam(10, 1, 500, 0.5, 0.9, 0.9, 0.0001, [], [], [], [], [], [])
# ii, ff, ssx, ssy, xx, yy = Adam(10, 1, 1000, 0.5, 1.1, 0.9, 0.0001, [], [], [], [], [], [])
# ii, ff, ssx, ssy, xx, yy = Adam(1, 1, 2000, 0.3, 1.1, 0.95, 0.0001, [], [], [], [], [], [])
# plt.plot(ii, ff, color='b', label='function value with alpha=0.1,beta1 = 0.25,beta2 = 0.25')
# plt.plot(ii1, ff1, color='r', label='function value with alpha=0.2,beta1 = 0.25,beta2 = 0.25')
# plt.plot(ii2, ff2, color='g', label='function value with alpha=0.2,beta1 = 0.5,beta2 = 0.25')
# plt.plot(ii3, ff3, color='y', label='function value with alpha=0.2,beta1 = 0.5,beta2 = 0.5')
# plt.plot(ii4, ff4, color='c', label='function value with alpha=0.3,beta1 = 0.9,beta2 = 0.9')
# plt.plot(ii5, ff5, color='m', label='function value with alpha=0.5,beta1 = 0.9,beta2 = 0.9')
# plt.plot(ii, ff, color='b', label='function value with alpha=0.5,beta1 = 1.1,beta2 = 0.9')
# plt.plot(ii, ssy, color='b', label='y step with alpha=0.1, beta = 0.25,beta2 = 0.25')
# plt.plot(ii1, ssy1, color='r', label='y step with alpha=0.2, beta = 0.25,beta2 = 0.25')
# plt.plot(ii2, ssy2, color='g', label='y step with alpha=0.2, beta = 0.5,beta2 = 0.25')
# plt.plot(ii3, ssy3, color='y', label='y step with alpha=0.2, beta = 0.5,beta2 = 0.5')
# plt.plot(ii4, ssy4, color='c', label='y step with alpha=0.2, beta = 0.9,beta2 = 0.9')
# plt.plot(ii5, ssy5, color='m', label='y step with alpha=0.5, beta = 0.9,beta2 = 0.9')
# x = np.arange(-5, 11, 0.1)
# y = np.arange(-1, 1.5, 0.1)
# X, Y = np.meshgrid(x, y)
# contours = plt.contour(X, Y, getPF1(X, Y), 20);
# plt.clabel(contours, inline=True, fontsize=10)
# plt.plot(xx, yy, color='b', label='the path of x and y with alpha=0.1, beta = 0.25,beta2 = 0.25')
# plt.plot(xx1, yy1, color='r', label='the path of x and y with alpha=0.2, beta = 0.25,beta2 = 0.25')
# plt.plot(xx2, yy2, color='g', label='the path of x and y with alpha=0.2, beta = 0.5,beta2 = 0.25')
# plt.plot(xx3, yy3, color='y', label='the path of x and y with alpha=0.2, beta = 0.5,beta2 = 0.5')
# plt.plot(xx4, yy4, color='c', label='the path of x and y with alpha=0.2, beta = 0.9,beta2 = 0.9')
# plt.plot(xx5, yy5, color='m', label='the path of x and y with alpha=0.5, beta = 0.9,beta2 = 0.9')
plt.xlabel('iteration')
plt.ylabel('function value of log')
plt.yscale("log")
# plt.xscale("log")
# plt.ylim((-0.5, 0.1))
plt.legend()
plt.show()
