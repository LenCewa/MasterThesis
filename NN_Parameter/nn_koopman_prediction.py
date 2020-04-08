import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jax import grad, jit, vmap
from jax import random
from sample import *
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

layer_sizes = [1, 17, 17, 1] # no. parameter in [350 , 400]
params = init_network_params(layer_sizes, random.PRNGKey(0))
initial_params = params
step_size = 1e-4
trajectory = get_sampled_trajectory('weakly_pendulum')
steps = len(trajectory) - 1 # andernfalls = 30

X = jnp.array(trajectory[0:steps]).reshape(-1,1) # train values
Y = jnp.array(trajectory[1:steps + 1]) # train labels


def relu(x):
    return jnp.maximum(0, x)
def tanh(x):
    return jnp.tanh(x)

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

# X_normal = jnp.array(trajectory[0:steps])
# true_y = Y - X_normal
# loss_values = []
#
# for i in range(1000000):
#     params = update(params, X, 100*true_y)
#     if i % 100 == 0:
#         l = loss(params, X, 100*true_y)
#         loss_values += [l]
#         print(i, l)
#
# execution_time = (time.time() - start_time) / 60
# print("Execution Time: ", execution_time)
#
# # Save Parameter
# np.save("debugged", params)
#
# # Save Loss
# np.save("/home/len/ReinforcementLearning/MasterThesis/NN_Loss/debugged", loss_values)

params = np.load("debugged.npy")

# Plot y and y_hat
# X_normal = jnp.array(trajectory[0:steps]) # Wegen Batch Predict haben wir einen Reshape bei der Variablen X
# true_y = Y - X_normal
# y_hat = []
#
# for i in range(steps):
#     y_hat += [predict(params, jnp.array([X_normal[i]]))/100]
#
# print("y_hat: ", y_hat)
# plt.figure()
# plt.plot(true_y, label="y_train")
# plt.plot(y_hat, label="f(x_train)")
# plt.legend()
# plt.show()
#
# print(jnp.sum(true_y)/len(true_y))
# print(y_hat)



# Plot NN Prediction
x0 = jnp.pi - 1
koopman_preds = [jnp.array([x0])]
#Roll-Out: x'_1 = x0 + f_NN(x0) then x'_2 = x'_1 + f(x'_1)
#True values
for i in range(steps):
    koopman_preds += [koopman_preds[i] + predict(params, koopman_preds[i])/100]

t = np.linspace(0, 20, num=500)
# Plot result
fig, ax = plt.subplots()
ax.plot(t, trajectory, label='x(t)')
ax.plot(t, koopman_preds, label='x\'_{t+1} = x\'_t + NN(x\'_t) and x_0 = π-10⁻²')
ax.set(xlabel='time (s)', ylabel='θ (rad)', title='Neural Network Dynamics Prediction')
ax.grid()
plt.legend()
#fig.savefig("NN.pdf")
plt.show()

# print("Koopman Preds", koopman_preds)
# plt.figure()
# plt.plot(koopman_preds, label='NN preds')
# plt.plot(trajectory[0:steps], label='trajectory')
# plt.ylim(-1,4)
# plt.legend()
# plt.show()