import math

import sympy
from matplotlib import pyplot as plt

x = sympy.symbols('x', real=True)
f1 = 0.5 * x ** 2
f2 = x ** 2
f3 = 2 * x ** 2
dfdx1 = sympy.diff(f1, x)
dfdx2 = sympy.diff(f2, x)
dfdx3 = sympy.diff(f3, x)
print(f1,"---->",dfdx1)
print(f2,"---->",dfdx2)
print(f3,"---->",dfdx3)

f1 = sympy.lambdify(x, f1)
f2 = sympy.lambdify(x, f2)
f3 = sympy.lambdify(x, f3)
dfdx1 = sympy.lambdify(x, dfdx1)
dfdx2 = sympy.lambdify(x, dfdx2)
dfdx3 = sympy.lambdify(x, dfdx3)
ii1 = []
ff1 = []
xx1 = []
ii2 = []
ff2 = []
xx2 = []
ii3 = []
ff3 = []
xx3 = []

def getF1(x):
    return f1(x)

def getF2(x):
    return f2(x)

def getF3(x):
    return f3(x)


def getDf1(x):
    return dfdx1(x)

def getDf2(x):
    return dfdx2(x)

def getDf3(x):
    return dfdx3(x)


def descent1(startX, alpha, maxIters):
    x = startX
    i = 0
    ii1.append(i)
    xx1.append(x)
    ff1.append(getF1(x))
    while i < maxIters:
        step = alpha * getDf1(x)
        x = x - step
        i = i + 1
        ii1.append(i)
        xx1.append(x)
        ff1.append(getF1(x))

def descent2(startX, alpha, maxIters):
    x = startX
    i = 0
    ii2.append(i)
    xx2.append(x)
    ff2.append(getF2(x))
    while i < maxIters:
        step = alpha * getDf2(x)
        x = x - step
        i = i + 1
        ii2.append(i)
        xx2.append(x)
        ff2.append(getF2(x))

def descent3(startX, alpha, maxIters):
    x = startX
    i = 0
    ii3.append(i)
    xx3.append(x)
    ff3.append(getF3(x))
    while i < maxIters:
        step = alpha * getDf3(x)
        x = x - step
        i = i + 1
        ii3.append(i)
        xx3.append(x)
        ff3.append(getF3(x))


descent1(2.0, 0.1, 50)
descent2(2.0, 0.1, 50)
descent3(2.0, 0.1, 50)
plt.plot(xx1, ff1, color='b', label='y(x) value with gamma=0.5')
plt.plot(xx2, ff2, color='r', label='y(x) value with gamma=1')
plt.plot(xx3, ff3, color='g', label='y(x) value with gamma=2')
plt.step(xx1, ff1, color='y', label='iterate steps gamma=0.5')
plt.step(xx2, ff2, color='c', label='iterate steps gamma=1')
plt.step(xx3, ff3, color='m', label='iterate steps gamma=2')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.legend()
plt.show()
