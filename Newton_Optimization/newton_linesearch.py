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

def newton_optimization_linesearch(y, x0, iterations, alpha0, damping0):
    res = [jnp.array([x0])]
    err = []
    alpha = alpha0
    damping = damping0
    roh = [1.2, 0.5, 1, 0.5, 0.01]

    for k in range(iterations):
        x = res[k]
        fx = fourier.predict(fourier.coefficients, x)
        dfx = dL(x,y)
        ddfx = ddL(x,y)
        i = 0

        err += [np.linalg.norm(y - fx)]
        if err[k] < 1e-3: break

        d = -dfx / (ddfx + damping)
        fx_alphad = fourier.predict(fourier.coefficients, x + alpha * d)
        while fx_alphad > (fx + roh[4]*dfx * alpha * d):
            print("Iteration: ", k , " while-loop: ", i)
            print("f(x + alpha * d) = ", fx_alphad, " > f(x) + r*f'(x) = ", fx + roh[4]*dfx)
            i += 1
            alpha = roh[1]*alpha
            # Optionally:
            damping = roh[2]*damping
            d = -dfx / (ddfx + damping)
            fx_alphad = fourier.predict(fourier.coefficients, x + alpha * d)

        x = x + alpha * d
        res += [x]
        alpha = np.min([roh[0], alpha, 1])

        # Optinally:
        damping = roh[3] * damping

    return res, err

t = jnp.linspace(0, 10*np.pi, num=1000)
x0 = 2
y0 = fourier.predict(fourier.coefficients, x0)
const_y0 = np.full(len(t), y0)

f = fourier.batched_predict(fourier.coefficients, t)
df = d_fourier.batched_predict(fourier.coefficients, t)
ddf = dd_fourier.batched_predict(fourier.coefficients, t)
const_0 = np.full(len(t), 0)

# Run Newton Optimization
steps = 20
x_start = 1.5
alpha0 = 1
damping0 = 0.999
res, err = newton_optimization_linesearch(y0[0], x_start, steps, alpha0, damping0)

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
fig.suptitle("Newton Line Search: x* = " + str(x0) + ", y* = " + str(y0[0]) + ", x0 = " + str(x_start) + " ||| steps = " + str(steps))
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
#plt.savefig("LineSearchFigs/Newton_LineSearch_x0=" + str(x_start) + ".png")

