import numpy as np
import matplotlib.pyplot as plt

def newton_optimization_method(y, x0, iterations):
    res = [x0]
    err = []

    for k in range(iterations):
        err += [np.linalg.norm(y - g(res[k]))]
        if err[k] < 1e-3: break
        x = res[k] - dL(res[k]) / ddL(res[k])
        res += [x]
    return res, err

def g(x):
    return x**2 + 1

def dL(x):
    return 4*x*(x**2+1)

def ddL(x):
    return 8*x**2 + 4*(x**2 + 1)

res, err = newton_optimization_method(0, 8, 20)
print("res", res)
print("err", err)


