#import gym
#import quanser_robots
#env = gym.make('Pendulum-v0')
#env.reset()
#env.render()

from Fourier import Fourier
from Fourier_Koopman_eigenfunctions import FourierKoopmanEigenfunctions
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sample import *
import json


# Hyperparameter
omega = 1
T = (2 * jnp.pi) / omega
step_size = 0.001
N = 10
iterations = 100

trajectory = get_sampled_trajectory('weakly_pendulum')
fke = FourierKoopmanEigenfunctions(T, omega, step_size, N, iterations, trajectory, 2)
loc = fke.compute_coefficients()  # loc = list of coeficcients
lops = [fke.batched_predict(c, trajectory[:-1]).ravel() for c in loc]  # lops = list of preds (prediction

# Plot result
plt.figure()
plt.plot(np.sin(trajectory), label="train")
plt.plot(lops[0], label="pred")
plt.legend()
plt.show()

# Save loc and lops
np.save("Koopman_Coefficients/" + "test_run_N=10_iterations=100", loc)
np.save("Koopman_Predictions/" + "test_run_N=10_iterations=100", lops)

'''
# Do computation
values, labels = trigeonmetric_product(-10, 10, 500)
fourier = Fourier(T, omega, step_size, N, iterations, values, labels)
coefficients = fourier.compute_coefficients()
preds = fourier.batched_predict(coefficients, values)[:, 0]

# Plot result
plt.figure()
plt.plot(labels, label="train")
plt.plot(preds, label="pred")
plt.legend()


# Timestamp for saving Fourier coefficients, hyperparameter and plots
dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Save plot
plt.savefig("Plots/" + dt + ".png")

# Save Fourier coefficients
np.save("Fourier_Coefficients/" + dt, coefficients)

# Save labels
np.save("Labels/" + dt, labels)

# Save preditctions
np.save("Predictions/" + dt, preds)

# Save values
np.save("Values/" + dt, values)

# Save hyperparameter
data = {}
data['hyperparameter'] = []
data['hyperparameter'].append({
    'omega': str(omega),
    'T': str(T),
    'step_size': str(step_size),
    'iterations': str(iterations)
})
with open("Hyperparameter/" + dt + ".txt", 'w') as outfile:
    json.dump(data, outfile)

plt.show()
'''