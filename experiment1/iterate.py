import math
import numpy as np

def fx(x):
    return x**5+x**4+math.exp(x)-11*x+1

def fax(x):
    return (x**5+x**4+math.exp(x)+1)/11

def iterate(x, lam):
    return x - lam * fx(x)

print('f(x)=x^5+x^4+e^x−11*x+1')
print('迭代函数fa(x)=(x^5+x^4+e^x+1)/11')
print('迭代函数2:fa2(x)=x-λf(x)')

xs = np.array([-2.0, 0.25, 1.5])
for x0 in xs:
    if x0==0.25:
        for i in range(100):
            x0 = fax(x0)
            if abs(fx(x0)) < 1e-10:
                print("Converged")
                print(fx(x0))
                print("iter:", i+1)
                print("root:", x0)
                break
    else:
        for j in range(100):
            x1 = iterate(x0, 0.01)
            if abs(x1-x0)<1e-10:
                print("Converged")
                print(fx(x1))
                print("iter:", j+1)
                print("root:", x1)
                break

            x0 = x1