import math
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

        print(t, "  x=", x, "  f(x)=", fx(x), "  delta=", delta)

        if abs(fx(x)) < 1e-10:
            print("Converged")
            break

    return x

print("x=0时,f(x)=" + str(fx(0)) + ">0")
print("x=1时,f(x)=" + str(fx(1)) + "<0")
print("所以在[0,1]区间内有根")
left=0
right=1
x0 = random.random() * (right - left) + left
print("Initial x:", x0)
x = hillClimbing(x0, left, right)

print(x)
print(fx(x))