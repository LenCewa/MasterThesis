import jax.numpy as jnp
from numpy import linalg as LA
import matplotlib.pyplot as plt
# https://arxiv.org/pdf/1709.01568.pdf for G and A
# Different approach: https://papers.nips.cc/paper/8138-deep-dynamical-modeling-and-control-of-unsteady-fluid-flows.pdf


def G_matrix(gX, dim):
    G = jnp.zeros((dim, dim))
    for k in range(len(gX)):
        #print("gX[", k, "]: ", gX[k].reshape(-1, 1))
        G += jnp.matmul(gX[k].reshape(-1, 1), gX[k].reshape(1, -1))
    return G


def A_matrix(gX, gY, dim):
    A = jnp.zeros((dim, dim))
    for k in range(len(gX)):
        A += jnp.matmul(gX[k].reshape(-1, 1), gY[k].reshape(1, -1))
    return A

def A_matrix_multistep(gX0, gY, dim):
    A = jnp.zeros((dim, dim))
    for k in range(len(gY)):
        A += jnp.matmul(gX0.reshape(-1, 1), gY[k].reshape(1, -1))
    return A

def pseudo_inverse(G, dim):
    # https://en.wikipedia.org/wiki/Singular_value_decomposition#Pseudoinverse
    UΣV = jnp.linalg.svd(G)
    pinv = jnp.matmul(jnp.matmul(UΣV[2].T, jnp.diag(1/UΣV[1])), UΣV[0].T)
    return pinv

def matrix_power(K, p):
    M = K
    for p in range(p - 1):
        M = M @ K
    return M

def plot_conditon(K, cond_history):
    cond = LA.cond(K)
    print("COND(K) = ", cond)
    # if (len(cond_history) % 100 == 0):
    #     plt.figure()
    #     plt.plot(cond_history)
    #     plt.show()
    return cond_history + [cond]

