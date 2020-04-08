import numpy as np
import matplotlib.pyplot as plt
from Fourier import *
from util import *

N = 5
steps = 500
dim = 12
basis_vector = 5  # max dim - 1

loc = np.load("/home/len/ReinforcementLearning/MasterThesis/Koopman_Coefficients/test_run_N=15_iterations=5000_dim=12.npy")
c = loc[basis_vector]
dN = 5
ddN = 5

omega = 1
T = (2 * jnp.pi) / omega
step_size = 0.001
iterations = 450

fourier = Fourier(T, omega, step_size, N, iterations, [], [])
d_fourier = dFourier(T, omega, step_size, N, iterations, [], [])
dd_fourier = ddFourier(T, omega, step_size, N, iterations, [], [])



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


def L(x, y):
    fx = fourier.predict(fourier.coefficients, x)
    return np.abs(y - fx)**2

def dL(x, y):
    fx = fourier.predict(fourier.coefficients, x)
    dfx = d_fourier.predict(fourier.coefficients, x)
    return 2 * (y - fx) * (-dfx)

def ddL(x, y):
    fx = fourier.predict(fourier.coefficients, x)
    dfx = d_fourier.predict(fourier.coefficients, x)
    ddfx = dd_fourier.predict(fourier.coefficients, x)
    return 2 * (dfx**2 - (y - fx) * ddfx)


t = jnp.linspace(0, 10*np.pi, num=1000)
x0 = 1
y0 = fourier.predict(fourier.coefficients, x0)
const_y0 = np.full(len(t), y0)

f = fourier.batched_predict(fourier.coefficients, t)
df = d_fourier.batched_predict(fourier.coefficients, t)
ddf = dd_fourier.batched_predict(fourier.coefficients, t)
const_0 = np.full(len(t), 0)

L = L(t, y0)
dL = dL(t, y0)
ddL = ddL(t, y0)

fig, axs = plt.subplots(3, 2)
axs[0, 0].plot(t, f)
axs[0, 0].plot(t, const_y0, 'tab:red')
axs[0, 0].set_title('f and y*')

axs[1, 0].plot(t, df, 'tab:orange')
axs[1, 0].set_title('df')

axs[2, 0].plot(t, ddf, 'tab:green')
axs[2, 0].set_title('ddf')

axs[0, 1].plot(t, L)
axs[0, 1].plot(t, const_0, 'tab:red')
axs[0, 1].set_title('L')

axs[1, 1].plot(t, dL, 'tab:orange')
axs[1, 1].plot(t, const_0, 'tab:red')
axs[1, 1].set_title('dL')

axs[2, 1].plot(t, ddL, 'tab:green')
axs[2, 1].set_title('ddL')

for ax in axs.flat:
    ax.set(xlabel='x-label', ylabel='y-label')

# Hide x labels and tick labels for top plots and y ticks for right plots.
# for ax in axs.flat:
#     ax.label_outer()

plt.show()














# res, err = newton_optimization_method(0.217977, 0, 20)
# print("res", res)
# print("res", jnp.array(res))
# print("err", err)

# plt.figure()
# plt.plot(t, fourier.batched_predict(fourier.coefficients, t))  # f(x)
# #plt.plot(t, d_fourier.batched_predict(fourier.coefficients, t))  # f'(x)
# #plt.plot(t, dd_fourier.batched_predict(fourier.coefficients, t))  # f''(x)
# plt.plot(t, np.full(len(t), y0))
# plt.show()