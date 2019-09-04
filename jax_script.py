#https://colab.research.google.com/github/google/jax/blob/master/notebooks/neural_network_and_data_loading.ipynb#scrollTo=-fmWA06xYE7d
#https://github.com/google/jax/blob/master/examples/resnet50.py
#https://colab.research.google.com/github/google/jax/blob/master/notebooks/quickstart.ipynb
#https://github.com/google/jax#mini-libraries

from __future__ import print_function, division, absolute_import
import jax.numpy as jnp
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
step_size = 0.1
num_epochs = 10
batch_size = 1
n_targets = 1
params = init_network_params(layer_sizes, random.PRNGKey(0))
initial_params = params


def relu(x):
  return jnp.maximum(0, x)


def predict(params, inputval):
  # Initialize activations
  activations = inputval
  for w, b in params[:-1]:
    outputs = jnp.dot(w, activations) + b
    activations = relu(outputs)

  final_w, final_b = params[-1]
  return jnp.dot(final_w, activations) + final_b

try:
  # Make a batched version of the `predict` function
  batched_predict = vmap(predict, in_axes=(None, 0))
except TypeError:
  print('Invalid shapes!')

def accuracy(params, inputvals, targets):
  return -1 # TODO implement accurarcy


def loss(params, input, target):
  print("loss::input ", input)
  print("loss::target ", target)
  preds = predict(params, input) #batched_predict(params, input)
  squared_loss = jnp.power(preds - target, 2)[0]
  return squared_loss


#@jit
def update(params, x, y):
  print("Params: ", params)
  grads = grad(loss, argnums=0)(params, x, y)
  print("Gradients: ", grads)
  #print("Params: ", params)
  result = []
  print("Zipped Grads: ", list(zip(params, grads)))
  for (w, b), (dw, db) in zip(params, grads):
    result += [(w - step_size * dw, b - step_size * db)]
  return result

training_batch_size = 100
training_generator = random.normal(random.PRNGKey(0), (training_batch_size, 2 * 1))
train_values = jnp.array(training_generator)
train_labels = jnp.array(jnp.ones(training_batch_size)) # constant function
data = zip(train_values, train_labels)
#print("data: ", list(data))

test_batch_size = 3
test_generator = random.normal(random.PRNGKey(1), (test_batch_size, 2 * 1))
test_values = jnp.array(test_generator)
#test_labels = jnp.array(jnp.ones(test_batch_size)) # constant function

for x, y in data:
  params = update(params, x, y)

print("DONE")

# Check if params changed

print("Initial Params: ", initial_params)
print("Updated Params: ", params)

# Test accuracy
print(test_values)
test_results = []
for x in test_values:
  test_results += [predict(params, x)]

print("Test results: ", test_results)