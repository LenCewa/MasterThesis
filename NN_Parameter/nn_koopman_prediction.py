import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jax import grad, jit, vmap
from jax import random
from sample import *
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

layer_sizes = [1, 17, 17, 1] # no. parameter in [350 , 400]
steps = 30
params = init_network_params(layer_sizes, random.PRNGKey(0))
initial_params = params
#print("INITIAL PARAMS: ", initial_params)
step_size = 0.001
trajectory = get_sampled_trajectory('weakly_pendulum')
X = jnp.array(trajectory[0:steps]).reshape(-1,1) # train values
Y = jnp.array(trajectory[1:steps + 1]) # train labels
loss_values = []

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

#@jit
def update(params, x, y):
    grads = grad(loss)(params, x, y)
    l = loss(params, x, y)
    global loss_values
    loss_values += [l]
    return [(w - step_size * dw, b - step_size * db) for (w, b), (dw, db) in zip(params, grads)]

# start_time = time.time()

# for i in range(500):
#     params = update(params, X, Y)
#
# execution_time = (time.time() - start_time) / 60
# print("Execution Time: ", execution_time)
#
# # Save Parameter
# np.save("NN_Parameter/" + "layers_1_17_17_1_iterations=500", params)
#
# # Save Loss
# np.save("NN_Loss/layers_1_17_17_1_iterations=500", loss_values)

params = np.load("layers_1_186_1_iterations=5000.npy")

# Plot y and y_hat
X_normal = jnp.array(trajectory[0:steps]) # Wegen Batch Predict haben wir einen Reshape bei der Variablen X
true_y = Y - X_normal
y_hat = []
for i in range(1):
    y_hat += [predict(params, X_normal[i])]
print("params: ", params)
print("y_hat: ", y_hat)
#plt.figure()
#plt.plot(true_y)
#plt.plot(y_hat)
#plt.show()



# # Plot NN Prediction
# x0 = jnp.pi - 1e-2
# koopman_preds = [jnp.array([x0])]
# #Roll-Out: x'_1 = x0 + f_NN(x0) then x'_2 = x'_1 + f(x'_1)
# #True values
# for i in range(steps):
#     #print(koopman_preds)
#     #print(predict(params, koopman_preds[i]))
#     koopman_preds += [koopman_preds[i] + predict(params, koopman_preds[i])]
#
# print("Params: ", params)
# print("Koopman Preds", koopman_preds)
# plt.figure()
# plt.plot(koopman_preds, label='NN preds')
# plt.plot(trajectory[0:steps], label='trajectory')
# plt.ylim(-1,4)
# plt.legend()
# plt.show()