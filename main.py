#import gym
#import quanser_robots
#env = gym.make('Pendulum-v0')
#env.reset()
#env.render()

from Fourier import Fourier
import jax.numpy as jnp
from jax import random
from scipy import signal

T = 2 * jnp.pi
omega = 1
step_size = 0.001
N = 20
iterations = 100

training_generator = random.normal(random.PRNGKey(0), (100, 1))
train_values = 10 * jnp.array(training_generator)
train_labels = [signal.square(x, 0.5) for x in train_values]
train_labels = jnp.array(train_labels)[:, 0]

test_generator = random.normal(random.PRNGKey(1), (100, 1))
test_values = 10 * jnp.array(test_generator)
test_labels = [signal.square(x, 0.5) for x in test_values]
test_labels = jnp.array(test_labels)[:, 0]

fourier = Fourier(T, omega, step_size, N, iterations, train_values, train_labels)

coefficients = fourier.compute_coefficients()
initial_coefficients = fourier.get_init_coeff()

print("Initial Parameter", initial_coefficients)
print("New Parameter", coefficients)
preds = fourier.batched_predict(coefficients, test_values)[:, 0]

print("Test_Labels: ", test_labels, " Predictions: ", preds)

print("jnp.square(preds - test_labels): ", jnp.sum(jnp.square(preds - test_labels)) / 100)