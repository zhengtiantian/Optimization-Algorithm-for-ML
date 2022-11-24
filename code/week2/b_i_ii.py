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

def getF(x):
    print(f(x))
    return f(x)

def getDf(x):
    return dfdx(x)


def descent(startX=1, alpha=0.1, maxIters=50):
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
plt.figure(figsize=(5, 5))
plt.plot(xx, ff, color='b', label='the value of y(x)')
plt.step(xx, ff, color='r', label='')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.legend()
plt.show()
