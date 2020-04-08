import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from Fourier import Fourier
from sample import *
from util import *


N = 6
steps = 500
dim = 10
basis_vector = 0 #max dim - 1
fourier = Fourier(1, 1, 0.001, N, 0, [], [])

def set_fourier_coefficients(c, N):
    for i in range(N + 1):
        # N = 15 impliziert [a0, [a1, b1], ..., [a15, b15]] also 16 Listenelemente
        fourier.coefficients[i] = jnp.array(c[i])

#loc = np.load("Koopman_Coefficients/test_run_N=15_iterations=10000_dim=15.npy") # basis_vector = 5
loc = np.load("Koopman_Coefficients/test_run_N=6_iterations=450_dim=10.npy") # basis_vector = 4
# #############
# #Pendulum Params
#loc = np.load("Koopman_Coefficients/pendulum_test_run_N=6_iterations=450_dim=10.npy")
# #############
gX, gY = [], []
x0 = np.pi #- 1e-2


trajectory = get_sampled_trajectory('weakly_pendulum')#get_sampled_trajectory('pendulum')#
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

def koopman_mpc_prediction(K, prediction_horizon, steps, dim_subspace, basis_vector):
    mpc = []
    pred = 0
    for s in range(0, steps, prediction_horizon):
        basis = set_basis(dim_subspace, trajectory[s])
        for mpc_step in range(prediction_horizon):
            for k in range(dim_subspace):
                pred += np.linalg.matrix_power(K, mpc_step)[:, basis_vector][k] * basis[k]
            mpc += [pred]
            pred = 0
    return np.array(mpc)

def koopman_mpc_rollout_prediction(K, prediction_horizon, steps, dim_subspace, basis_vector):
    mpc_rollout = []
    pred = 0
    for s in range(steps):
        basis = set_basis(dim_subspace, trajectory[s])
        for mpc_step in range(prediction_horizon):
            for k in range(dim_subspace):
                pred += np.linalg.matrix_power(K, mpc_step)[:, basis_vector][k] * basis[k]
            mpc_rollout += [pred]
            pred = 0
    return np.array(mpc_rollout)

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
    X = np.linspace(0, np.pi*2, num=500)
    a0 = basis[0]
    for x in X:
        periodic_sum = 0
        for n in range(basis.shape[0] - 1):
            a_n = basis[n + 1][0]
            b_n = basis[n + 1][1]
            periodic_sum += a_n * jnp.cos((n + 1) * 1 * x) + b_n * jnp.sin((n + 1) * 1 * x)
        Y += [a0 / 2 + periodic_sum]
    return Y

'''
koopman_preds = koopman_prediction(K, x0, steps, dim, basis_vector)
# mpc1 = koopman_mpc_prediction(K, 1, steps, dim, basis_vector)
mpc5 = koopman_mpc_prediction(K, 5, steps, dim, basis_vector)
# mpc10= koopman_mpc_prediction(K, 10, steps, dim, basis_vector)
#mpc25= koopman_mpc_prediction(K, 25, steps, dim, basis_vector)
#mpc50= koopman_mpc_prediction(K, 50, steps, dim, basis_vector)
lifted_trajectory = lift_trajectory(trajectory, loc[basis_vector], steps)

t = np.linspace(0, 20, num=500)
# Plot result
fig, ax = plt.subplots()
ax.plot(t, lifted_trajectory, label='g_'+str(basis_vector)+'(x(t))')
ax.plot(t, koopman_preds, label='[K^t]g_'+str(basis_vector)+'(x_0)')
# ax.plot(t, mpc1, label='[K^t]g_'+str(basis_vector)+'(x_j) for a 1 time-step')
ax.plot(t, mpc5, label='[K^t]g_'+str(basis_vector)+'(x_j) for a 5 time-step')
# ax.plot(t, mpc10, label='[K^t]g_'+str(basis_vector)+'(x_j) for a 10 time-step')
#ax.plot(t, mpc25, label='[K^t]g_'+str(basis_vector)+'(x_j) for a 25 time-step')
#ax.plot(t, mpc50, label='[K^t]g_'+str(basis_vector)+'(x_j) for a 50 time-step')
ax.set(xlabel='time (s)', ylabel='g_'+str(basis_vector)+'(θ) (rad)', title='Koopman operator w.r.t. to different prediction horizons')
ax.grid()
plt.legend()
fig.savefig("EvalKoopmanBasis="+str(basis_vector)+"mpc2050INF.png")
plt.show()
'''


# plt.plot(koopman_preds, label='koopman preds')
# plt.plot(lifted_trajectory, label='lifted trajectory')
# plt.plot(mpc, label='mpc')
# #plt.plot(embedding_function(loc[basis_vector]), label="embedding fucntion")
# plt.legend()
# plt.show()