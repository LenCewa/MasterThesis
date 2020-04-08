import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import matplotlib.pyplot as plt
import time

start_time = time.time()
# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1e-2):
  w_key, b_key = random.split(key)
  return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
  keys = random.split(key, len(sizes))
  return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

layer_sizes = [1, 17, 17, 1]
params = init_network_params(layer_sizes, random.PRNGKey(0))
initial_params = params
step_size = 1e-4

training_generator = jnp.linspace(0, 2*jnp.pi, num=50)
train_values = jnp.array(training_generator)
train_labels = [jnp.sin(x) for x in train_values]
train_labels = jnp.array(train_labels)
train_values = jnp.array(training_generator).reshape(-1,1)

test_generator = jnp.linspace(0, 2*jnp.pi, num=500)
test_values = jnp.array(test_generator)
test_labels = [jnp.sin(x) for x in test_values]
test_labels = jnp.array(test_labels)
test_values = jnp.array(test_generator).reshape(-1,1)

def relu(x):
    return jnp.maximum(0, x)

def predict(params, inputs):
    activations = inputs
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = relu(outputs)

    final_w, final_b = params[-1]
    result = jnp.dot(final_w, activations) + final_b
    return result

batched_predict = vmap(predict, in_axes=(None, 0))

def loss(params, train_values, train_labels):
    preds = batched_predict(params, train_values)[:, 0]
    return jnp.sum(jnp.square(preds - train_labels))

@jit
def update(params, x, y):
    grads = grad(loss)(params, x, y)
    return [(w - step_size * dw, b - step_size * db) for (w, b), (dw, db) in zip(params, grads)]

loss_values = []

for i in range(1000000):
    params = update(params, train_values, train_labels)
    if i % 100 == 0:
        l = loss(params, train_values, train_labels)
        loss_values += [l]
        print(i, l)

execution_time = (time.time() - start_time) / 60
print("Execution Time: ", execution_time)
print(jnp.sum(test_labels)/len(test_labels))
print(batched_predict(params, test_values)[0, 0])
plt.figure()
plt.plot(batched_predict(params, test_values)[:, 0], "r.-",   label="prediction")
plt.plot(test_labels, "b.-", label="labels")
plt.legend()
plt.show()
