import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

from scipy import signal

# https://www.mathematik.tu-darmstadt.de/media/analysis/lehrmaterial_anapde/hallerd/M2InfSkript16.pdf
T = 2 * jnp.pi
omega = 1
step_size = 0.001
N = 20

# A helper function to randomly initialize Fourier coefficients for a Fourier polynomial
def random_fourier_params(key, scale=1e-2):
    key1, key2 = random.split(key)
    return scale * random.normal(key2, (2,))

def init_fourier_polynomial(N, key):
    '''
    Initialize all Fourier parameters for a Fourier polynomial of degree N
    :param N: Degree of the Fourier polynomial
    :return: {float32, list of DeviceArrays} a0 the initial bias, a list of DeviceArrays with parameter [(a1, b1), (a2, b2), ..., (aN, bN)]
    '''
    a_0 = random.normal(key, (1,))
    keys = random.split(key, N)
    return [a_0] + [random_fourier_params(k) for k in keys]

def predict(coefficients, x):
    periodic_sum = 0
    a_0 = coefficients[0]
    for n in range(len(coefficients) - 1):
        a_n = coefficients[n+1][0]
        b_n = coefficients[n+1][1]
        periodic_sum += a_n * jnp.cos((n + 1) * omega * x) + b_n * jnp.sin((n + 1) * omega * x)
    return a_0/2 + periodic_sum

batched_predict = vmap(predict, in_axes=(None, 0))

def loss(coefficents, train_values, train_labels):
    preds = batched_predict(coefficents, train_values)[:, 0]
    return jnp.sum(jnp.square(preds - train_labels))

@jit
def update(coefficients, x, y):
    grads = grad(loss)(coefficients, x, y)
    a_0 = coefficients[0]
    da_0 = grads[0]
    #print("Coefficients: ", coefficients)
    #print("Grads: ", grads)
    #print("ZIP: ", list(zip(coefficients[1:], grads[1:])))
    #print("UPDATE: ", [a_0 - step_size * da_0] + [coeff - step_size * dcoeff for coeff, dcoeff in zip(coefficients[1:], grads[1:])]) # [(a_n - step_size * da_n, b_n - step_size * db_n) for (a_n, b_n), (da_n, db_n) in zip(coefficients[1:], grads[1:])]
    return [a_0 - step_size * da_0] + [coeff - step_size * dcoeff for coeff, dcoeff in zip(coefficients[1:], grads[1:])]

coefficients = init_fourier_polynomial(N, random.PRNGKey(0))
initial_coefficients = coefficients

training_generator = random.normal(random.PRNGKey(0), (100000, 1))
train_values = 10 * jnp.array(training_generator)
train_labels = [signal.square(x, 0.5) for x in train_values]
train_labels = jnp.array(train_labels)[:, 0]

test_generator = random.normal(random.PRNGKey(1), (5, 1))
test_values = 10 * jnp.array(test_generator)
test_labels = [signal.square(x, 0.5) for x in test_values]
test_labels = jnp.array(test_labels)[:, 0]

for i in range(1000):
    coefficients = update(coefficients, train_values, train_labels)

print("Initial Parameter", initial_coefficients)
print("New Parameter", coefficients)
preds = batched_predict(coefficients, test_values)[:, 0]

print("jnp.square(preds - test_labels): ", jnp.square(preds - test_labels))