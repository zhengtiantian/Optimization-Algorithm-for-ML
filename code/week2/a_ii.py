import sympy
from matplotlib import pyplot as plt

x = sympy.symbols('x', real=True)
f = x ** 4
dfdx = sympy.diff(f, x)
f = sympy.lambdify(x, f)
dfdx = sympy.lambdify(x, dfdx)
x = -0.1
delta = 0.01

xx = []
d = []
ed = []
while x < 0.1:
    xx.append(x)
    d.append(dfdx(x))
    ed.append(((x + delta) ** 4 - x ** 4) / delta)
    x = x + 0.01

plt.plot(xx, d, color='b', label='actual derivative values')
plt.plot(xx, ed, color='r', label='estimate derivative values')
plt.xlabel('x')
plt.ylabel('devirative values')
plt.legend()
plt.show()
