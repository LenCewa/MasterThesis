import jax.numpy as jnp
from Fourier import Fourier
from jax import grad, jit
from functools import partial
from util import *

# TODO: Coefficients should be a list [g_1, g_2, ..., g_{self.dim_sub_space}]

class FourierKoopmanEigenfunctions(Fourier):
    def __init__(self, T, omega, step_size, N, iterations, X_trajectory, Y_trajectory, dim_sub_space):
        Fourier.__init__(self, T, omega, step_size, N, iterations, X_trajectory, Y_trajectory)
        self.dim_sub_space = dim_sub_space

    def KoopmanLoss(self):
        '''
        Computes the Koopman-loss for given Fourier-polynomials, where Ψ := g = [g_1, ..., g_{self.dim_sub_space}] is a vector of Fourier-polynomials of degree N
        :return: jnp.sum[(g(x_2) - K¹ * g(x_1))² + ... + (g(x_n) - K^{n-1} * g(x_1))²]
        '''
        return -1

    # https://github.com/google/jax/issues/1251
    @partial(jit, static_argnums=(0,))
    def update(self, coefficients, x, y):
        grads = grad(self.KoopmanLoss())(coefficients, x, y)
        a_0 = coefficients[0]
        da_0 = grads[0]
        return [a_0 - self.step_size * da_0] + [coeff - self.step_size * dcoeff for coeff, dcoeff in zip(coefficients[1:], grads[1:])]

    def compute_coefficients(self):
        # TODO: Iterate over list of coefficents which has length = self.dim_sub_space
        coeffs = self.coefficients
        for i in range(self.iterations):
            coeffs = self.update(coeffs, self.tv, self.tl)
        return coeffs