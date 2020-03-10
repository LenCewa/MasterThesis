import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from Fourier import *
from util import *

N = 15
steps = 500
dim = 15
basis_vector = 5  # max dim - 1
dN = 5  # Cutoff
ddN = 5  # Cutoff

fourier = Fourier(1, 1, 0, N, 0, [], [])
d_fourier = dFourier(1, 1, 0, dN, 0, [], [])
dd_fourier = ddFourier(1, 1, 0, ddN, 0, [], [])

loc = np.load("/home/len/ReinforcementLearning/MasterThesis/Koopman_Coefficients/test_run_N=15_iterations=10000_dim=15.npy")
c = loc[basis_vector]

def set_fourier_coefficients(c, N):
    for i in range(N + 1):
        # N = 15 impliziert [a0, [a1, b1], ..., [a15, b15]] also 16 Listenelemente
        fourier.coefficients[i] = jnp.array(c[i])


def set_dfourier_coefficients(c, N):
    for i in range(N + 1):
        # N = 15 impliziert [a0, [a1, b1], ..., [a15, b15]] also 16 Listenelemente
        d_fourier.coefficients[i] = jnp.array(c[i])


def set_ddfourier_coefficients(c, N):
    for i in range(N + 1):
        # N = 15 impliziert [a0, [a1, b1], ..., [a15, b15]] also 16 Listenelemente
        dd_fourier.coefficients[i] = jnp.array(c[i])


# Testing purpose: x0 = 0, x = 7, y = 0.217
def newton_optimization_method(y, x0, iterations):
    res = [jnp.array([x0])]
    err = []

    for k in range(iterations):
        set_fourier_coefficients(c, N)
        set_dfourier_coefficients(c, dN)
        set_ddfourier_coefficients(c, ddN)
        g_x = fourier.predict(fourier.coefficients, res[k])
        dg_x = d_fourier.predict(d_fourier.coefficients, res[k])
        ddg_x = dd_fourier.predict(dd_fourier.coefficients, res[k])

        err += [np.linalg.norm(y - g_x)]

        if err[k] < 1e-3: break
        dL = 2 * (y - g_x) * (-dg_x)
        ddL = 2 * (dg_x**2 - (y - g_x) * ddg_x)

        #print("dL = ", dL)
        #print("ddL = ", ddL)
        x = res[k] - dL / ddL
        res += [x]

    return res, err

t = jnp.linspace(0, 20, num=500)
res, err = newton_optimization_method(0.217, 0, 20)
print("res", res)
print("res", jnp.array(res))
print("err", err)


plt.figure()
plt.plot(fourier.batched_predict(fourier.coefficients, jnp.array(res)))
plt.plot(fourier.batched_predict(fourier.coefficients, t[:21]))
plt.show()