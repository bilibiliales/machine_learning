import math
import numpy as np
import random

def fx(x):
    return x**5+x**4+math.exp(x)-11*x+1

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

        if abs(fx(x)) < 1e-10:
            print("Converged")
            break

    return x

areas=np.array([[-3,-2],[0,1],[1,2]])
for area in areas:
    left=area[0]
    right=area[1]
    x0 = random.random() * (right - left) + left
    print("Initial x:", x0)
    x = hillClimbing(x0, left, right)

    print(x)
    print(fx(x))