import numpy as np
import matplotlib.pyplot as plt

loss = np.load("pendulum_test_run_N=6_iterations=450_dim=10.npy")

plt.figure()
plt.plot(loss)
plt.show()