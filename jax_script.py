#https://colab.research.google.com/github/google/jax/blob/master/notebooks/neural_network_and_data_loading.ipynb#scrollTo=-fmWA06xYE7d
#https://github.com/google/jax/blob/master/examples/resnet50.py
#https://colab.research.google.com/github/google/jax/blob/master/notebooks/quickstart.ipynb
#https://github.com/google/jax#mini-libraries

from __future__ import print_function, division, absolute_import
import jax.numpy as np
from jax import grad, jit, vmap
from jax import random
import time

# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1e-2):
  w_key, b_key = random.split(key)
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


def relu(x):
  return np.maximum(0, x)


def predict(params, inputval):
  # per-example predictions
  print("Predict the value of (x1,x2): ", inputval)

  # Initialize activations
  activations = inputval
  for w, b in params[:-1]:
    outputs = np.dot(w, activations) + b
    activations = relu(outputs)

  final_w, final_b = params[-1]
  return np.dot(final_w, activations) + final_b

try:
  # Make a batched version of the `predict` function
  batched_predict = vmap(predict, in_axes=(None, 0))
except TypeError:
  print('Invalid shapes!')

def accuracy(params, inputvals, targets):
  return -1 # TODO implement accurarcy


def loss(params, input, target):
  preds = batched_predict(params, input)
  return np.abs(preds) - np.abs(target)


@jit
def update(params, x, y):
  grads = grad(loss)(params, x, y)
  return [(w - step_size * dw, b - step_size * db)
          for (w, b), (dw, db) in zip(params, grads)]

training_generator = random.normal(random.PRNGKey(0), (10, 2 * 1))
train_values = np.array(training_generator)
train_labels = np.array(np.ones(10)) # constant function

test_generator = random.normal(random.PRNGKey(1), (10, 2 * 1))
test_values = np.array(test_generator)
test_labels = np.array(np.ones(10)) # constant function

for epoch in range(num_epochs):
  start_time = time.time()
  for x, y in training_generator:
    params = update(params, x, y)
  epoch_time = time.time() - start_time

print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))