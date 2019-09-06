from jax import random

key = random.PRNGKey(0)
print(random.normal(key, shape=(3,)))
print(random.normal(key, shape=(3,)))

key = random.PRNGKey(0)

key, subkey = random.split(key)
print(random.normal(subkey, shape=(3,)))  # [ 1.1378783  -1.22095478 -0.59153646]

key, subkey = random.split(key)
print(random.normal(subkey, shape=(3,)))  # [-0.06607265  0.16676566  1.17800343]