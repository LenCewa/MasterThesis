import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jax import grad, jit, vmap
from jax import random
from sample import *

# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1e-2):
  w_key, b_key = random.split(key)
  return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key):
  keys = random.split(key, len(sizes))
  return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

layer_sizes = [1, 186, 1] # no parameter in [350 , 400]
steps = 30
params = init_network_params(layer_sizes, random.PRNGKey(0))
initial_params = params
#print("INITIAL PARAMS: ", initial_params)
step_size = 0.001
trajectory = get_sampled_trajectory('weakly_pendulum')
X = jnp.array(trajectory[0:steps]).reshape(-1,1) # train values
Y = jnp.array(trajectory[1:steps + 1]) # train labels

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
    #print("Loss: ", jnp.sum(jnp.square(preds - train_labels)))
    return jnp.sum(jnp.square(preds - train_labels))

@jit
def update(params, x, y):
    grads = grad(loss)(params, x, y)
    #print(grads)
    return [(w - step_size * dw, b - step_size * db) for (w, b), (dw, db) in zip(params, grads)]

# for i in range(5000):
#     params = update(params, X, Y)
#     preds = batched_predict(params, X)
#     #print(params)
#     #print("Loss = ", jnp.sum(jnp.square(preds[:, 0] - Y)), " || in iteration = ", i + 1)
#
# # Save Parameter
# np.save("NN_Parameter/" + "layers_1_186_1_iterations=5000", params)

params = np.load("NN_Parameter/layers_1_186_1_iterations=5000.npy")
# Plot NN Prediction
x0 = jnp.pi - 1e-2
koopman_preds = [jnp.array([x0])]
for i in range(steps):
    #print(koopman_preds)
    #print(predict(params, koopman_preds[i]))
    koopman_preds += [predict(params, koopman_preds[i])]

print("Params: ", params)
print("Koopman Preds", koopman_preds)
plt.plot(koopman_preds, label='NN preds')
plt.plot(trajectory[0:steps], label='trajectory')
plt.legend()
plt.show()

