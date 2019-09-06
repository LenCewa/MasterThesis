import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

# https://www.mathematik.tu-darmstadt.de/media/analysis/lehrmaterial_anapde/hallerd/M2InfSkript16.pdf
# Let T = 2pi and omega = 1

T = 2 * jnp.pi
omega = 1



def init_fourier_polynomial(N):
    '''
    A helper function to randomly initialize Fourier coefficients for a Fourier polynomial
    :param N: Degree of the Fourier polynomial
    :return: {float32, list} a0 the initial bias, a list of tuples with parameter [(a1, b1), (a2, b2), ..., (aN, bN)]
    '''


    return -1