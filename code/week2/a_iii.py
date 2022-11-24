import sympy
from matplotlib import pyplot as plt

x = sympy.symbols('x', real=True)
f = x ** 4
dfdx = sympy.diff(f, x)
f = sympy.lambdify(x, f)
dfdx = sympy.lambdify(x, dfdx)
x = -0.5
delta0 = 1
delta1 = 0.1
delta2 = 0.01
delta3 = 0.001

xx = []
d = []
ed0 = []
ed1 = []
ed2 = []
ed3 = []
while x < 0.5:
    xx.append(x)
    d.append(dfdx(x))
    ed0.append(((x + delta0) ** 4 - x ** 4) / delta0)
    ed1.append(((x + delta1) ** 4 - x ** 4) / delta1)
    ed2.append(((x + delta2) ** 4 - x ** 4) / delta2)
    ed3.append(((x + delta3) ** 4 - x ** 4) / delta3)
    x = x + 0.01

plt.plot(xx, d, color='b', label='actual derivative values')
plt.plot(xx, ed0, color='r', label='estimate derivative values delta = 1')
plt.plot(xx, ed1, color='y', label='estimate derivative values delta = 0.1')
plt.plot(xx, ed2, color='g', label='estimate derivative values delta = 0.01')
plt.plot(xx, ed3, color='black', label='estimate derivative values delta = 0.001')
plt.xlabel('x')
plt.ylabel('devirative values')
plt.legend()
plt.show()
