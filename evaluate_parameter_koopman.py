import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from Fourier import Fourier
from sample import *
from util import *


N = 15
steps = 500
dim = 15
basis_vector = 5 #max dim - 1
fourier = Fourier(1, 1, 0, N, 0, [], [])

def set_fourier_coefficients(c, N):
    for i in range(N + 1):
        # N = 15 impliziert [a0, [a1, b1], ..., [a15, b15]] also 16 Listenelemente
        fourier.coefficients[i] = jnp.array(c[i])

loc = np.load("Koopman_Coefficients/test_run_N=15_iterations=10000_dim=15.npy")
gX, gY = [], []
x0 = np.pi - 1e-2


trajectory = get_sampled_trajectory('weakly_pendulum')
P = len(trajectory)
print(P)
X = trajectory[:-1]
Y = trajectory[1:]
print("Shape::::::::::::.... ", loc[0].shape)

for c in loc:
    set_fourier_coefficients(c, N)
    gX += [fourier.batched_predict(fourier.coefficients, X).ravel()]
    gY += [fourier.batched_predict(fourier.coefficients, Y).ravel()]

gX = jnp.array(gX).T
gY = jnp.array(gY).T

A = 1 / P * A_matrix(gX, gY, dim)
G = 1 / P * G_matrix(gX, dim)

G = jnp.linalg.inv(G)  # pseudo_inverse(G, self.dim)
K = jnp.matmul(G, A)

#print("Koopman Operator: ", K)
print("Koopman Operator shape: ", K.shape)



'''
2. n-Step prediction 
'''


def set_basis(dim_subspace, x0):
    basis = []
    for k in range(dim_subspace):
        a0 = loc[k][0]
        periodic_sum = 0
        for n in range(loc[k].shape[0] - 1):
            a_n = loc[k][n+1][0]
            b_n = loc[k][n+1][1]
            periodic_sum += a_n * jnp.cos((n + 1) * 1 * x0) + b_n * jnp.sin((n + 1) * 1 * x0)

        basis += [a0/2 + periodic_sum]
    return basis


def koopman_prediction(K, x0, steps, dim_subspace, basis_vector):
    '''

    :param K:
    :param x0: start value
    :param steps:
    :param dim_subspace:
    :param basis_vector:
    :return:
    '''
    basis = set_basis(dim_subspace, x0)
    koopman_preds = []
    pred = 0
    for s in range(steps):
        for k in range(dim_subspace):
            pred += np.linalg.matrix_power(K, s)[:, basis_vector][k] * basis[k]
        koopman_preds += [pred]
        pred = 0
    return np.array(koopman_preds)


def lift_trajectory(trajectory, basis, steps):
    lifted_trajectory = []
    a0 = basis[0]
    for x in trajectory[:steps]:
        periodic_sum = 0
        for n in range(basis.shape[0] - 1):
            a_n = basis[n + 1][0]
            b_n = basis[n + 1][1]
            periodic_sum += a_n * jnp.cos((n + 1) * 1 * x) + b_n * jnp.sin((n + 1) * 1 * x)
        lifted_trajectory += [a0 / 2 + periodic_sum]
    return lifted_trajectory

def embedding_function(basis):
    Y = []
    X = np.linspace(0, 20, num=500)
    a0 = basis[0]
    for x in X:
        periodic_sum = 0
        for n in range(basis.shape[0] - 1):
            a_n = basis[n + 1][0]
            b_n = basis[n + 1][1]
            periodic_sum += a_n * jnp.cos((n + 1) * 1 * x) + b_n * jnp.sin((n + 1) * 1 * x)
        Y += [a0 / 2 + periodic_sum]
    return Y

koopman_preds = koopman_prediction(K, x0, steps, dim, basis_vector)
lifted_trajectory = lift_trajectory(trajectory, loc[basis_vector], steps)

plt.plot(koopman_preds, label='koopman preds')
plt.plot(lifted_trajectory, label='lifted trajectory')
#plt.plot(embedding_function(loc[basis_vector]), label="embedding fucntion")
plt.legend()
plt.show()