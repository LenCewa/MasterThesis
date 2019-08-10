import numpy as np
from jax import vmap

a = np.array([1, 2, 3])
m1 = np.array([[1, 1, 1], [2, 2, 2]])

def dotProduct(u, v):
    return np.vdot(u, v)

mv = vmap(dotProduct, (0, None), 0)

print(mv(m1, a))
