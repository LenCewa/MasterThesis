import matplotlib.pyplot as plt
import numpy as np
from sample import *


reconstruct = np.load("mpc_rollout_basis=0.npy")
trajectory = get_sampled_trajectory('weakly_pendulum')
#re = np.load("Reconstruction_Errors/second_try_max_iter=10_tol=1e-3.npy")
t = np.linspace(0, 20, num=500)
# Plot result
print(reconstruct[:6])
fig, ax = plt.subplots()
ax.plot(t, trajectory, label='x(t)')
ax.plot(t, reconstruct, label='{argmin_t ||Ψ(x_t)-y_n||² : t = 1, ..., 20}')
ax.set(xlabel='time (s)', ylabel='θ (rad)', title='Recovering the original trajectory from the Koopman predictions')
ax.grid()
plt.legend()
fig.savefig("mpcRolloutBasis=0.png")
plt.show()
