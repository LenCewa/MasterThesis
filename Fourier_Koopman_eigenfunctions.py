import jax.numpy as jnp
import numpy as np
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
        self.coefficients = [self.init_fourier_polynomial(N, random.PRNGKey(i)) for i in range(self.dim)]
        #self.n_batched_predict = vmap(self.batched_predict, in_axes=(0, None))

    def cond_K(self, loc, X, Y, cond_history):
        gX, gY = [], []
        for c in loc:
            gX += [self.batched_predict(c, X).ravel()]
            gY += [self.batched_predict(c, Y).ravel()]

        gX = jnp.array(gX).T
        gY = jnp.array(gY).T

        A = 1 / self.P * A_matrix(gX, gY, self.dim)
        G = 1 / self.P * G_matrix(gX, self.dim)

        G = jnp.linalg.inv(G)  # pseudo_inverse(G, self.dim)
        K = jnp.matmul(G, A)
        cond_history = plot_conditon(K, cond_history)
        return cond_history

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

        A = 1 / self.P * A_matrix(gX, gY, self.dim)
        G = 1 / self.P * G_matrix(gX, self.dim)

        G = jnp.linalg.inv(G)  # pseudo_inverse(G, self.dim)
        K = jnp.matmul(G, A)
        print("gX.shape = {} | gY.shape = {} | K.shape = {}".format(gX.shape, gY.shape, K.shape))
        #print("NN loss: ", jnp.sum(jnp.square(gY - jnp.dot(gX, K))))
        print("Analytic Loss: ", jnp.sum(jnp.square(gY - jnp.dot(gX, K))))
        return jnp.sum(jnp.square(gY - jnp.dot(gX, K)))

    def multiple_step_Koopman_loss(self, loc, X, Y):
        gX0, gY = [], []
        for c in loc:
            gX0 += [self.predict(c, X[0]).ravel()]
            gY += [self.batched_predict(c, Y).ravel()]

        gX0 = jnp.array(gX0).T
        print("gX0: ", gX0)
        gY = jnp.array(gY).T

        A = 1 / self.P * A_matrix_multistep(gX0, gY, self.dim)
        G = G_matrix(gX0, self.dim)

        G = jnp.linalg.inv(G)  # pseudo_inverse(G, self.dim)
        K = jnp.matmul(G, A)

        loss = 0
        for p in range(15):#range(len(gY)):
            #print("K = ", K, " pow: ", p, " K^p: ", matrix_power(K, p))
            loss += jnp.sum(jnp.square(gY - jnp.dot(gX0, matrix_power(K, p))))
        print("Analytic Loss: ", loss)
        return loss

    #@partial(jit, static_argnums=(0,))
    def update(self, loc, X, Y, cond_history):
        cond_history = self.cond_K(loc, X, Y, cond_history)
        grads = grad(self.simple_Koopman_loss)(loc, X, Y)
        updated_loc = []
        for d in range(self.dim):
            print("Gradients[", d, "]: ", grads[d])
            a_0 = loc[d][0]
            da_0 = grads[d][0]
            updated_loc += [[a_0 - self.step_size * da_0] + [coeff - self.step_size * dcoeff for coeff, dcoeff in zip(loc[d][1:], grads[d][1:])]]
        return updated_loc, cond_history

    def compute_coefficients(self):
        loc = self.coefficients
        cond_history = []
        for i in range(self.iterations):
            print("Iteration: ", i + 1, "/", self.iterations, " START:")
            loc, cond_history = self.update(loc, self.X_trajectory, self.Y_trajectory, cond_history)
            print("List of coefficients: ", loc)
            #print("Condition(K): ", cond_history[-1])
        return loc, cond_history


