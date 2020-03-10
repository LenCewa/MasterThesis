from Fourier import Fourier, dFourier, ddFourier
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from sample import *

# Hyperparameter
omega = 1
T = (2 * np.pi) / omega
step_size = 0.001
N = 5
iterations = 1000

ddfourier = ddFourier(T, omega, step_size, N, iterations, [], [])
dfourier = dFourier(T, omega, step_size, N, iterations, [], [])
fourier = Fourier(T, omega, step_size, N, iterations, [], [])
#loc = np.load("Fourier_Coefficients/2019-09-13 17:44:06.npy") # Random Fourier
#loc = np.load("Fourier_Coefficients/2019-09-13 19:47:39.npy") # trigenometric product
loc = np.load("Fourier_Coefficients/2020-03-04 12:41:13_N=15.npy") #trigeonmetric product
print(type(loc[0][0]))
def set_fourier_coefficients(c, N):
    for i in range(N):
        # N = 15 impliziert [a0, [a1, b1], ..., [a15, b15]] also 16 Listenelemente
        ddfourier.coefficients[i] = jnp.array(c[i])

set_fourier_coefficients(loc, N)

values, labels = random_fourier(-10, 10, 500)
preds = ddfourier.batched_predict(ddfourier.coefficients, values)#[:, 0]

x = np.linspace(-10,10,500)

plt.figure()
plt.plot(preds)
#Trigonometric Product
plt.plot(2*np.sin(x)**3 -7*np.cos(x)**2*np.sin(x)) # second order derivative
#plt.plot(np.cos(x)**3 - 2*np.cos(x)*np.sin(x)**2) # first order derivative

#Random Fourier 5 * jnp.sin(x) + 2 * jnp.cos(4 * x)
#plt.plot(-5*np.sin(x) - 32*np.cos(4*x)) # second order derivative
#plt.plot(5*np.cos(x) - 8*np.sin(4*x)) # first order derivative

#plt.plot(labels) # original function
plt.show()