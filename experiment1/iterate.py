import math

def fx(x):
    return x**5+x**4+math.exp(x)-11*x+1

def fax(x):
    return (x**5+x**4+math.exp(x)+1)/11

print('f(x)=x^5+x^4+e^x−11*x+1')
print('迭代函数x=(x^5+x^4+e^x+1)/11')

x = 0
for i in range(100):
    x = fax(x)
    print(str(i + 1) + "     x=" + str(x) + "     f(x)=" + str(fx(x)))
    if abs(fx(x))<1e-10:
        print("Converged")
        break
print(x)
print(fx(x))