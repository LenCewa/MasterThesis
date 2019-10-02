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
N = 6
iterations = 100


trajectory = get_sampled_trajectory('weakly_pendulum')
#fke = FourierKoopmanEigenfunctions(T, omega, step_size, N, iterations, trajectory, 12)
#loc = fke.compute_coefficients()  # loc = list of coeficcients
#lops = [fke.batched_predict(c, trajectory[:-1]).ravel() for c in loc]  # lops = list of preds (predictions)

lops = np.load("Koopman_Predictions/test_run_N=8_iterations=10.npy")

fig, ax1 = plt.subplots()
ax1.set_xlabel("steps")
ax1.set_ylabel("train label")
ax1.plot(np.sin(trajectory), label="sin(trajectory)")
'''ax2 = ax1.twinx()
ax3 = ax1.twinx()
ax4 = ax1.twinx()
ax5 = ax1.twinx()
ax6 = ax1.twinx()'''
ax7 = ax1.twinx()
#ax2.set_ylabel("Koopman prediction")
'''ax2.plot(lops[0], "g", label="lops[0]")
ax3.plot(lops[1], "r", label="lops[1]")
ax4.plot(lops[2], "c", label="lops[2]")
ax5.plot(lops[3], "m", label="lops[3]")
ax6.plot(lops[4], "y", label="lops[4]")'''
ax7.plot(lops[5], "k", label="lops[5]")
fig.tight_layout()
fig.legend()
plt.show()

'''# Plot result
plt.figure()
plt.plot(np.sin(trajectory), label="train")
plt.plot(lops[0], label="pred")
plt.legend()
plt.show()'''

# Save loc and lops
np.save("Koopman_Coefficients/" + "test_run_N=8_iterations=10", loc)
np.save("Koopman_Predictions/" + "test_run_N=8_iterations=10", lops)

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