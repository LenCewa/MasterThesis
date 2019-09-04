import jax.numpy as jnp
from jax import vmap

a = jnp.array([[1, 2, 3]]).transpose()
b = jnp.array([1, 2, 3])
m1 = jnp.array([[1, 2, 1], [2, 2, 2]])
m2 = jnp.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

vv = lambda x, y: jnp.vdot(x, y)
mv1 = vmap(vv, (0, None), 0)
mv2 = vmap(vv, (None, 0), 0)
mm1 = vmap(mv1, (None, 1), 1) #(1, 0) (None, 1)
mm2 = vmap(mv2, (1, 0), 0)

#print("mv1", mv1(m1, b))
#print("mv2", mv2(m1, b))
print("mm1", mm1(m1, m2))
#print("mm2", mm2(m1, a))

