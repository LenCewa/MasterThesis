import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from functools import partial

# https://www.mathematik.tu-darmstadt.de/media/analysis/lehrmaterial_anapde/hallerd/M2InfSkript16.pdf
class Fourier(object):
    def __init__(self, T, omega, step_size, N, iterations, train_values, train_labels):
        self.T = T
        self.omega = omega
        self.step_size = step_size
        self.N = N
        self.iterations = iterations
        self.tv = train_values
        self.tl = train_labels
        self.coefficients = self.init_fourier_polynomial(self.N, random.PRNGKey(0))
        self.batched_predict = vmap(self.predict, in_axes=(None, 0))

    # A helper function to randomly initialize Fourier coefficients for a Fourier polynomial
    def random_fourier_params(self, key, scale=1e-2):
        key1, key2 = random.split(key)
        return scale * random.normal(key2, (2,))

    def init_fourier_polynomial(self, N, key):
        '''
        Initialize all Fourier parameters for a Fourier polynomial of degree N
        :param N: Degree of the Fourier polynomial
        :return: {float32, list of DeviceArrays} a0 the initial bias, a list of DeviceArrays with parameter [(a1, b1), (a2, b2), ..., (aN, bN)]
        '''
        a_0 = random.normal(key, (1,))
        keys = random.split(key, N)
        return [a_0] + [self.random_fourier_params(k) for k in keys]

    def predict(self, coefficients, x):
        periodic_sum = 0
        a_0 = coefficients[0]
        for n in range(len(coefficients) - 1):
            a_n = coefficients[n+1][0]
            b_n = coefficients[n+1][1]
            periodic_sum += a_n * jnp.cos((n + 1) * self.omega * x) + b_n * jnp.sin((n + 1) * self.omega * x)
        return a_0/2 + periodic_sum

    def loss(self, coefficents, train_values, train_labels):
        preds = self.batched_predict(coefficents, train_values)[:, 0]
        return jnp.sum(jnp.square(preds - train_labels))

    # https://github.com/google/jax/issues/1251
    @partial(jit, static_argnums=(0,))
    def update(self, coefficients, x, y):
        grads = grad(self.loss)(coefficients, x, y)
        a_0 = coefficients[0]
        da_0 = grads[0]
        return [a_0 - self.step_size * da_0] + [coeff - self.step_size * dcoeff for coeff, dcoeff in zip(coefficients[1:], grads[1:])]

    def compute_coefficients(self):
        coeffs = self.coefficients
        #print((coeffs))
        for i in range(self.iterations):
            coeffs = self.update(coeffs, self.tv, self.tl)
        return coeffs

    def get_initial_coefficients(self):
        return self.coefficients