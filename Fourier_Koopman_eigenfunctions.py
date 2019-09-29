import jax.numpy as jnp
from Fourier import Fourier
from jax import grad, jit, vmap
from jax import random
from functools import partial
from util import *

class FourierKoopmanEigenfunctions(Fourier):
    def __init__(self, T, omega, step_size, N, iterations, trajectory, dim_sub_space):
        '''

        :param T:
        :param omega:
        :param step_size:
        :param N:
        :param iterations:
        :param trajectory: {numpy n-d-array}
        :param dim_sub_space:
        '''
        self.trajectory = trajectory
        self.P = len(trajectory)
        self.X_trajectory = trajectory[:-1]
        self.Y_trajectory = trajectory[1:]
        self.dim = dim_sub_space
        Fourier.__init__(self, T, omega, step_size, N, iterations, self.X_trajectory, self.Y_trajectory)
        self.coefficients = [self.init_fourier_polynomial(N, random.PRNGKey(0)) for _ in range(self.dim)]
        #self.n_batched_predict = vmap(self.batched_predict, in_axes=(0, None))

    def simple_Koopman_loss(self, loc, X, Y):
        '''
        Computes the Koopman-loss for given Fourier-polynomials, where Ψ := g = [g_1, ..., g_{self.dim_sub_space}] is a vector of Fourier-polynomials of degree N
        :param loc: {} list of coefficients
        :param X: {} X_trajectory (train values)
        :param Y: {} Y_trajectory (train labels)
        :return: jnp.sum[(g(x_2) - g(x_1) * K¹)² + ... + (g(x_n) - g(x_1) * K^{n-1})²]
        '''
        gX, gY = [], []
        for c in loc:
            gX += [self.batched_predict(c, X).ravel()]
            gY += [self.batched_predict(c, Y).ravel()]

        gX = jnp.array(gX).T
        gY = jnp.array(gY).T

        A = 1/(self.P) * A_matrix(gX, gY, self.dim)
        G = 1/(self.P) * G_matrix(gX, self.dim)

        # Not working:
        #G = jnp.linalg.pinv(G)
        #K = jnp.linalg.lstsq(A, G) (oder .lstsq(G, A) - nochmal genau überlegen)

        G = pseudo_inverse(G, self.dim)
        K = jnp.matmul(G, A) #G @ A
        return jnp.sum(jnp.square(gY - jnp.dot(gX, K)))

    #@partial(jit, static_argnums=(0,))
    def update(self, loc, X, Y):
        grads = grad(self.simple_Koopman_loss)(loc, X, Y)
        updated_loc = []
        for d in range(self.dim):
            a_0 = loc[d][0]
            da_0 = grads[d][0]
            updated_loc += [[a_0 - self.step_size * da_0] + [coeff - self.step_size * dcoeff for coeff, dcoeff in zip(loc[d][1:], grads[d][1:])]]
        return updated_loc

    def compute_coefficients(self):
        loc = self.coefficients
        for i in range(self.iterations):
            loc = self.update(loc, self.X_trajectory, self.Y_trajectory)
            print("Iteration: ", i, " done.")
        return loc
