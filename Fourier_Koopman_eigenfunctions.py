import jax.numpy as jnp
from Fourier import Fourier
from jax import grad, jit
from functools import partial
from util import *

# TODO:
#  1. BLush Koopman Loss where we have K^{m} for some natural number m
#  2. One X_trajectory vs n different X_trajectories. And how would I compute the loss? Should I average over the different computed losses?
#  3. How does "coefficients" look like if Ψ := g = [g_1, g_2]?

class FourierKoopmanEigenfunctions(Fourier):
    def __init__(self, T, omega, step_size, N, iterations, X_trajectory, Y_trajectory, loss_step_size):
        Fourier.__init__(self, T, omega, step_size, N, iterations, X_trajectory, Y_trajectory)
        self.loss_step_size = loss_step_size

    def KoopmanLoss(self):
        '''
        Computes the loss_step_size-Koopman-loss for given Fourier-polynomials, where Ψ := g = [g_1, g_2] is a vector of Fourier-polynomials of degree N
        :return: 0.5 * [(g(x_2) - K^{self.loss_step_size} * g(x_1))² + ... + (g(x_n) - K^{self.loss_step_size} * g(x_{n-1}))²]
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
        coeffs = self.coefficients
        for i in range(self.iterations):
            coeffs = self.update(coeffs, self.tv, self.tl)
        return coeffs