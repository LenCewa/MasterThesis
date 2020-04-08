import numpy as np
import matplotlib.pyplot as plt
from Fourier import *
from util import *


# Init Params for Fourier-Classes
N = 5
omega = 1
T = (2 * jnp.pi) / omega
step_size = 0.001
iterations = 450

fourier = Fourier(T, omega, step_size, N, iterations, [], [])
d_fourier = dFourier(T, omega, step_size, N, iterations, [], [])
dd_fourier = ddFourier(T, omega, step_size, N, iterations, [], [])


def compute_beta(y, t):
    N = len(fourier.coefficients)
    A = 0 #np.abs(fourier.coefficients(0))
    for n in range(N - 1):
        a_n = fourier.coefficients[n + 1][0]
        b_n = fourier.coefficients[n + 1][1]
        A += np.abs(a_n) + np.abs(b_n)
    A = A*N
    B = A*N**2
    f = fourier.batched_predict(fourier.coefficients, t)
    d = np.max(np.abs(f-y))

    return 2*(A + d * B)

def compute_gamma(no_steps):
    l = [0]
    g = []
    for s in range(2 * no_steps):
        l += [(1+np.sqrt(1+4*l[s]**2))/2]
        g += [(1-l[s])/l[s+1]]
    return g[1:], l

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

def nesterov_step(y0, x0, no_steps, gamma_list, beta):
    res = [jnp.array([x0])]
    y_list = res
    err = []

    for k in range(no_steps):
        x = res[k]
        fx = fourier.predict(fourier.coefficients, x)
        err += [np.linalg.norm(y0 - fx)]
        if err[k] < 1e-3: break
        dfx = d_fourier.predict(fourier.coefficients, x)
        y = x - dfx[0] / beta
        y_list += [y]

        x = (1 - gamma_list[k])*y + gamma_list[k]*y_list[k]
        res += [x]

    return res, err

t = jnp.linspace(0, 10*np.pi, num=1000)
x0 = 2
y0 = fourier.predict(fourier.coefficients, x0)
const_y0 = np.full(len(t), y0)

f = fourier.batched_predict(fourier.coefficients, t)
df = d_fourier.batched_predict(fourier.coefficients, t)
ddf = dd_fourier.batched_predict(fourier.coefficients, t)
const_0 = np.full(len(t), 0)

# Run Nestrov Gradient Descent
beta = compute_beta(y0, t)
steps = 1000
x_start = 1
gamma_list, l = compute_gamma(steps)
res, err = nesterov_step(y0[0], x_start, steps, gamma_list, beta)

fx_t = []
ex_t = []
for x in res:
    pred = fourier.predict(fourier.coefficients, x)[0]
    fx_t += [pred]
    ex_t += [(y0 - pred)**2]

print(res)
print(err)

L = L(t, y0)
dL = dL(t, y0)
ddL = ddL(t, y0)

fig, axs = plt.subplots(3, 2)
fig.suptitle("Nestorv: x* = " + str(x0) + ", y* = " + str(y0[0]) + ", x0 = " + str(x_start) + " ||| steps = " + str(steps))
axs[0, 0].plot(t, f)
axs[0, 0].plot(t, const_y0, 'tab:red')
axs[0, 0].plot(res, fx_t, 'k.-')
axs[0, 0].plot(res[-2], fx_t[-2], 'ro')
axs[0, 0].plot(res[-1], fx_t[-1], 'g*')
axs[0, 0].set_title('f and y*')

axs[1, 0].plot(t, df, 'tab:orange')
axs[1, 0].set_title('df')

axs[2, 0].plot(t, ddf, 'tab:green')
axs[2, 0].set_title('ddf')

axs[0, 1].plot(t, L)
axs[0, 1].plot(t, const_0, 'tab:red')
axs[0, 1].plot(res, ex_t, 'k.-')
axs[0, 1].plot(res[-2], ex_t[-2], 'ro')
axs[0, 1].plot(res[-1], ex_t[-1], 'g*')
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

