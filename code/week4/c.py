
import sympy
import math

from matplotlib import pyplot as plt
from sympy import Max

x = sympy.symbols('x', real=True)
f = Max(x,0)
dfdx = sympy.diff(f, x)
print(dfdx)
f = sympy.lambdify(x, f)
dfdx = sympy.lambdify(x, dfdx)


def getF1(x):
    return f(x)


def getDf1(x):
    return dfdx(x)


def getDf(x):
    if x > 5:
        return 1
    else:
        return 0


def RMSProp(startX, startY, maxIters, alpha0, Beta, epsilon, ii, ff, ssx, ssy, xx, yy):
    x = startX
    i = 0
    sum = 0
    while i < maxIters:
        ii.append(i)
        ff.append(getF1(x))
        xx.append(x)
        sum = Beta * sum + (1 - Beta) * getDf1(x) ** 2
        alpha = alpha0 / (math.sqrt(sum) + epsilon)

        x = x - alpha * getDf1(x)
        i = i + 1
        ssx.append(alpha)



    return ii, ff, ssx, ssy, xx, yy


def heavyBall(startX, startY, maxIters, alpha, Beta, ii, ff, ssx, ssy, xx, yy):
    x = startX
    i = 0
    z = 0
    while i < maxIters:
        ii.append(i)
        ff.append(getF1(x))
        z = Beta * z + alpha * getDf1(x)
        x = x - z
        i = i + 1
        ssx.append(z)
        xx.append(x)


    return ii, ff, ssx, ssy, xx, yy


def Adam(startX, startY, maxIters, alpha, Beta1, Beta2, epsilon, ii, ff, ssx, ssy, xx, yy):
    x = startX
    i = 0
    mt = 0
    vt = 0
    while i < maxIters:
        ii.append(i)
        ff.append(getF1(x))
        mt = Beta1 * mt + (1 - Beta1) * getDf1(x)
        print("mt:" + str(mt))
        vt = Beta2 * vt + (1 - Beta2) * (getDf1(x) ** 2)
        print("vt:" + str(vt))
        me = mt / (1 - Beta1 ** (i + 1))
        print("me:"+str(me))
        ve = vt / (1 - Beta2 ** (i + 1))
        print("ve:" + str(ve))
        x = x - alpha * (me / (math.sqrt(ve) + epsilon))
        i = i + 1
        ssx.append(alpha * (me / (math.sqrt(ve) + epsilon)))
        xx.append(x)

    return ii, ff, ssx, ssy, xx, yy

plt.figure(figsize=(7, 7))

# ii, ff, ssx, ssy, xx, yy = RMSProp(1, None, 100, 0.01, 0.9, 0.0001, [], [], [], [], [], [])
# ii1, ff1, ssx1, ssy1, xx1, yy1 = RMSProp(1, None, 100, 0.05, 0.25, 0.0001, [], [], [], [], [], [])
# ii2, ff2, ssx2, ssy2, xx2, yy2 = RMSProp(1, None, 100, 0.05, 0.9, 0.0001, [], [], [], [], [], [])
# ii3, ff3, ssx3, ssy3, xx3, yy3 = RMSProp(1, None, 100, 0.1, 0.9, 0.0001, [], [], [], [], [], [])
# ii, ff, ssx, ssy, xx, yy = heavyBall(100, None, 1000, 0.1, 0.25, [], [], [], [], [], [])
# ii1, ff1, ssx1, ssy1, xx1, yy1 = heavyBall(100, None, 1000, 0.5, 0.25, [], [], [], [], [], [])
# ii2, ff2, ssx2, ssy2, xx2, yy2 = heavyBall(100, None, 1000, 0.5, 0.9, [], [], [], [], [], [])
# ii3, ff3, ssx3, ssy3, xx3, yy3 = heavyBall(100, None, 1000, 0.9, 0.9, [], [], [], [], [], [])
# ii, ff, ssx, ssy, xx, yy = heavyBall(100, None, 1000, 0.1, 0.25, [], [], [], [], [], [])
# ii1, ff1, ssx1, ssy1, xx1, yy1 = heavyBall(100, None, 1000, 0.5, 0.25, [], [], [], [], [], [])
# ii2, ff2, ssx2, ssy2, xx2, yy2 = heavyBall(100, None, 1000, 0.5, 0.9, [], [], [], [], [], [])
# ii3, ff3, ssx3, ssy3, xx3, yy3 = heavyBall(100, None, 1000, 0.9, 0.9, [], [], [], [], [], [])
ii, ff, ssx, ssy, xx, yy = Adam(100, None,1000 , 0.1, 0.25, 0.25, 0.0001, [], [], [], [], [], [])
ii1, ff1, ssx1, ssy1, xx1, yy1 = Adam(100, None, 1000, 0.5, 0.25, 0.25, 0.0001, [], [], [], [], [], [])
ii2, ff2, ssx2, ssy2, xx2, yy2 = Adam(100, None, 1000, 0.5, 0.9, 0.25, 0.0001, [], [], [], [], [], [])
ii3, ff3, ssx3, ssy3, xx3, yy3 = Adam(100, None, 1000, 0.5, 0.9, 0.9, 0.0001, [], [], [], [], [], [])
ii4, ff4, ssx4, ssy4, xx4, yy4 = Adam(100, None, 1000, 0.9, 0.9, 0.9, 0.0001, [], [], [], [], [], [])

# plt.plot(ii, ff, color='b', label='function value with alpha=0.1, beta1 = 0.25')
# plt.plot(ii1, ff1, color='r', label='function value with alpha=0.5, beta = 0.25')
# plt.plot(ii2, ff2, color='g', label='function value with alpha=0.5, beta = 0.9')
# plt.plot(ii3, ff3, color='y', label='function value with alpha=0.9, beta = 0.9')
# plt.plot(ii, xx, color='b', label='function value with alpha=0.01, beta1 = 0.25')
# plt.plot(ii1, xx1, color='r', label='function value with alpha=0.05, beta = 0.25')
# plt.plot(ii2, xx2, color='g', label='function value with alpha=0.05, beta = 0.9')
# plt.plot(ii3, xx3, color='y', label='function value with alpha=0.1, beta = 0.9')
# plt.plot(ii, ff, color='b', label='function value with alpha=0.01, beta1 = 0.25')
# plt.plot(ii1, ff1, color='r', label='function value with alpha=0.05, beta = 0.25')
# plt.plot(ii2, ff2, color='g', label='function value with alpha=0.05, beta = 0.9')
# plt.plot(ii3, ff3, color='y', label='function value with alpha=0.1, beta = 0.9')
# plt.plot(ii, xx, color='b', label='function value with alpha=0.01, beta1 = 0.25')
# plt.plot(ii1, xx1, color='r', label='function value with alpha=0.05, beta = 0.25')
# plt.plot(ii2, xx2, color='g', label='function value with alpha=0.05, beta = 0.9')
# plt.plot(ii3, xx3, color='y', label='function value with alpha=0.1, beta = 0.9')
# plt.plot(ii, ff, color='b', label='function value with alpha=0.1, beta1 = 0.25')
# plt.plot(ii1, ff1, color='r', label='function value with alpha=0.5, beta = 0.25')
# plt.plot(ii2, ff2, color='g', label='function value with alpha=0.5, beta = 0.9')
# plt.plot(ii3, ff3, color='y', label='function value with alpha=0.9, beta = 0.9')
# plt.plot(ii, ssx, color='b', label='x step size with alpha=0.06, beta = 0.25')
# plt.plot(ii, ssy, color='r', label='y step size with alpha=0.06, beta = 0.25')
plt.plot(ii, ff, color='b', label='function value with alpha=0.1, beta1 = 0.25,beta2=0.25')
plt.plot(ii1, ff1, color='r', label='function value with alpha=0.5, beta1 = 0.25,beta2=0.25')
plt.plot(ii2, ff2, color='g', label='function value with alpha=0.5, beta1 = 0.9,beta2=0.25')
plt.plot(ii3, ff3, color='y', label='function value with alpha=0.5, beta1 = 0.9,beta2=0.9')
plt.plot(ii4, ff4, color='c', label='function value with alpha=0.9, beta1 = 0.9,beta2=0.9')
# plt.plot(ii, xx, color='b', label='function value with alpha=0.01, beta1 = 0.25,beta2=0.25')
# plt.plot(ii1, xx1, color='r', label='function value with alpha=0.05, beta1 = 0.25,beta2=0.25')
# plt.plot(ii2, xx2, color='g', label='function value with alpha=0.05, beta1 = 0.9,beta2=0.25')
# plt.plot(ii3, xx3, color='y', label='function value with alpha=0.05, beta1 = 0.9,beta2=0.9')
# plt.plot(ii4, xx4, color='c', label='function value with alpha=0.1, beta1 = 0.9,beta2=0.9')


plt.xlabel('iteration')
plt.ylabel('function value')
# plt.yscale("log")
# plt.xscale("log")
# plt.ylim((-0.5, 0.1))
plt.legend()
plt.show()
