import math

import sympy
from matplotlib import pyplot as plt

x = sympy.symbols('x', real=True)
f = x ** 4
dfdx = sympy.diff(f, x)
f = sympy.lambdify(x, f)
dfdx = sympy.lambdify(x, dfdx)
ii = []
ff = []
xx = []
ff0 = []
xx0 = []
ii1 = []
ff1 = []
xx1 = []
ii2 = []
ff2 = []
xx2 = []


def getF(x):
    # print(f(x))
    return f(x)


def getDf(x):
    return dfdx(x)


def original(maxIters):
    s = 1
    x = s;
    xx0.append(x)
    ff0.append(getF(x))
    i = 0
    while i < maxIters:
        x = x - (s / 50)
        xx0.append(x)
        ff0.append(getF(x))
        i = i + 1


def descent2(startX, alpha, maxIters):
    x = startX
    i = 0
    ii2.append(i)
    xx2.append(x)
    ff2.append(getF(x))
    while i < maxIters:
        step = alpha * getDf(x)
        x = x - step
        i = i + 1
        ii2.append(i)
        xx2.append(x)
        ff2.append(getF(x))


def descent1(startX, alpha, maxIters):
    x = startX
    i = 0
    ii1.append(i)
    xx1.append(x)
    ff1.append(getF(x))
    while i < maxIters:
        step = alpha * getDf(x)
        x = x - step
        i = i + 1
        ii1.append(i)
        xx1.append(x)
        ff1.append(getF(x))


def descent(startX, alpha, maxIters):
    x = startX
    i = 0
    ii.append(i)
    xx.append(x)
    ff.append(getF(x))
    while i < maxIters:
        step = alpha * getDf(x)
        x = x - step

        i = i + 1
        ii.append(i)
        xx.append(x)
        ff.append(getF(x))
        # ff.append(math.log10(getF(x)))


descent(1.0, 0.1, 50)
descent1(2.0, 0.02, 50)
descent2(1.0, 0.05, 50)
original(50)
plt.figure(figsize=(5, 5))
plt.plot(xx0, ff0, color='b', label='primitive function')
plt.plot(xx, ff, color='black', label='the value of y(x) x=1 alpha=0.1')
plt.plot(xx2, ff2, color='r', label='the value of y(x) x=1 alpha=0.05')
plt.step(xx, ff, color='y', label='iterate steps alpha=0.1')
plt.step(xx2, ff2, color='g', label='iterate steps alpha=0.05')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.legend()
plt.show()
