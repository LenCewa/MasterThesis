import jax.numpy as np
from jax import random
from jax.experimental import stax
from jax.experimental.stax import Conv, Dense, MaxPool, Relu, Flatten, LogSoftmax

# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1e-2):
  w_key, b_key = random.split(key)
  print("weight ", random.normal(w_key, (n, m)).shape)
  print("bias ", random.normal(b_key, (n,)).shape)
  return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
  keys = random.split(key, len(sizes))
  return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

layer_sizes = [2, 4, 4, 1]
param_scale = 0.1
step_size = 0.0001
num_epochs = 10
batch_size = 1
n_targets = 1
params = init_network_params(layer_sizes, random.PRNGKey(0))

print(params)

#https://colab.research.google.com/github/google/jax/blob/master/notebooks/neural_network_and_data_loading.ipynb#scrollTo=-fmWA06xYE7d
#https://github.com/google/jax/blob/master/examples/resnet50.py
#https://github.com/google/jax#mini-libraries