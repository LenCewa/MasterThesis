import jax.numpy as jnp
# https://arxiv.org/pdf/1709.01568.pdf for G and A
# Different approach: https://papers.nips.cc/paper/8138-deep-dynamical-modeling-and-control-of-unsteady-fluid-flows.pdf

def G_matrix(gX, dim):
    G = jnp.zeros((dim, dim))
    for k in range(len(gX)):
        G += gX[k].reshape(-1, 1) @ gX[k].reshape(1, -1)
    return G

def A_matrix(gX, gY, dim):
    A = jnp.zeros((dim, dim))
    for k in range(len(gX)):
        A += gX[k].reshape(-1, 1) @ gY[k].reshape(1, -1)
    return A

def pseudo_inverse(G, dim):
    # pinv(G) = (G'G)⁻¹G' (that's it?)
    return jnp.ones((dim, dim))