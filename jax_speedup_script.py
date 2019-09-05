import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1e-2):
  w_key, b_key = random.split(key)
  return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
  keys = random.split(key, len(sizes))
  return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

layer_sizes = [2, 3, 1]
params = init_network_params(layer_sizes, random.PRNGKey(0))
initial_params = params
step_size = 0.0001

training_generator = random.normal(random.PRNGKey(0), (100, 2))
train_values = jnp.array(training_generator)
train_labels = [a * b for a, b in train_values]
train_labels = jnp.array(train_labels)

test_generator = random.normal(random.PRNGKey(1), (5, 2 * 1))
test_values = jnp.array(test_generator)
test_labels = [a * b for a, b in test_values]
test_labels = jnp.array(test_labels)

def relu(x):
    return jnp.maximum(0, x)

def predict(params, inputs):
    activations = inputs
    for w, b in params:
        outputs = jnp.dot(w, activations) + b
        activations = relu(outputs)
    return outputs

batched_predict = vmap(predict, in_axes=(None, 0))

def loss(params, train_values, train_labels):
    preds = batched_predict(params, train_values)[:, 0]
    return jnp.sum(jnp.square(preds - train_labels))

@jit
def update(params, x, y):
    grads = grad(loss)(params, x, y)
    return [(w - step_size * dw, b - step_size * db) for (w, b), (dw, db) in zip(params, grads)]

for i in range(1000000):
    params = update(params, train_values, train_labels)

print("Initial Parameter", initial_params)
print("New Parameter", params)
preds = batched_predict(params, test_values)
print("jnp.square(preds[:,0] - test_labels): ", jnp.square(preds[:, 0] - test_labels))
