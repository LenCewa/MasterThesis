from Fourier import Fourier, dFourier, ddFourier
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from sample import *

# Hyperparameter
omega = 1
T = (2 * np.pi) / omega
step_size = 0.001
N = 25
iterations = 1000

ddfourier = ddFourier(T, omega, step_size, N, iterations, [], [])
dfourier = dFourier(T, omega, step_size, N, iterations, [], [])
fourier = Fourier(T, omega, step_size, N, iterations, [], [])
loc = np.load("Fourier_Coefficients/2019-09-13 17:44:06.npy") # Random Fourier
#loc = np.load("Fourier_Coefficients/2019-09-13 19:47:39.npy") # trigenometric product
#loc = np.load("Fourier_Coefficients/2020-03-04 12:41:13_N=15.npy") #trigeonmetric product
print(loc.shape)
def set_fourier_coefficients(c, N):
    for i in range(N):
        # N = 15 impliziert [a0, [a1, b1], ..., [a15, b15]] also 16 Listenelemente
        dfourier.coefficients[i] = jnp.array(c[i])

set_fourier_coefficients(loc, N)

x, y = random_fourier(-10, 10, 500)
y = 5*np.cos(x) - 8*np.sin(4*x)
#x, y = trigeonmetric_product(-10, 10, 500)
preds = dfourier.batched_predict(dfourier.coefficients, x)#[:, 0]

print("N = ", N)
print(y.reshape(-1,1).shape)
print(preds.reshape(-1,1).shape)
print("RMSE = ", np.sum(np.sqrt((y.reshape(-1,1)-preds.reshape(-1,1))**2)))

# Original Function
# fig, ax = plt.subplots()
# ax.plot(x, y, label='f(x) = 5sin(x) + 2cos(4x) + 7')
# ax.plot(x, preds, label='Fourier approx. of f')
# ax.set(xlabel='x', ylabel='y', title='Fourier approximation of 5sin(x) + 2cos(4x) + 7 and N = 15')
# ax.grid()
# plt.legend()
# fig.savefig("fourierapprox15.pdf")
# plt.show()

# First Order Derivative
fig, ax = plt.subplots()
ax.plot(x, 5*np.cos(x) - 8*np.sin(4*x), label='df(x) = 5cos(x) - 8sin(4x)')
ax.plot(x, preds, label='Fourier approx. of df')
ax.set(xlabel='x', ylabel='y', title='Fourier approximation of 5cos(x) - 8sin(4x) and N = 25')
ax.grid()
plt.legend()
fig.savefig("dfourierapprox25.pdf")
plt.show()

# Second Order Derivative
# fig, ax = plt.subplots()
# ax.plot(x, -5*np.sin(x) - 32*np.cos(4*x), label='d²f(x) = -5sin(x) - 32cos(4x)')
# ax.plot(x, preds, label='Fourier approx. of d²f')
# ax.set(xlabel='x', ylabel='y', title='Fourier approximation of -5sin(x) - 32cos(4x) and N = 25')
# ax.grid()
# plt.legend()
# fig.savefig("ddfourierapprox25.pdf")
# plt.show()

# plt.figure()
# plt.plot(x, preds)
# #Trigonometric Product
# plt.plot(x, 2*np.sin(x)**3 -7*np.cos(x)**2*np.sin(x)) # second order derivative
# #plt.plot(np.cos(x)**3 - 2*np.cos(x)*np.sin(x)**2) # first order derivative
#
# #Random Fourier 5 * jnp.sin(x) + 2 * jnp.cos(4 * x)
# #plt.plot(-5*np.sin(x) - 32*np.cos(4*x)) # second order derivative
# #plt.plot(5*np.cos(x) - 8*np.sin(4*x)) # first order derivative
#
# #plt.plot(labels) # original function
# plt.show()