import math
import numpy as np
import random
import matplotlib.pyplot as plt

def fx(x):
    return x**5+x**4+math.exp(x)-11*x+1

def show():
    x = np.linspace(-2.5, 2.5, 100)
    y = np.array([fx(i) for i in x])
    y0 = np.array([0 for i in x])
    dy_dx = np.gradient(y, x)
    plt.plot(x, dy_dx, label='f\'(x)')
    plt.plot(x, y, label='f(x)')
    plt.plot(x, y0, label='y=0')
    plt.legend()
    plt.show()

def hillClimbing(x, left, right):
    delta = 0.01
    t = 0

    while t < 999:
        t += 1

        x_left = max(left, x - delta)
        x_right = min(right, x + delta)

        xs = [x, x_left, x_right]
        best_x = min(xs, key=lambda v: abs(fx(v)))

        if abs(fx(best_x)) < abs(fx(x)):
            x = best_x
        else:
            delta /= 2

        #print(t, "  x=", x, "  f(x)=", fx(x), "  delta=", delta)

        if abs(fx(x)) < 1e-10:
            print("Converged")
            break

    return x
show()

areas=np.array([[-3,-2],[0,1],[1,2]])
for area in areas:
    left=area[0]
    right=area[1]
    x0 = random.random() * (right - left) + left
    print("Initial x:", x0)
    x = hillClimbing(x0, left, right)

    print(x)
    print(fx(x))