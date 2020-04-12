import matplotlib.pyplot as plt
import numpy as np

from Dynamical_Systems import weakly_pendulum

# trajectory of the weakly pendulum
y = weakly_pendulum.y

# p1 = np.load("mpc_rollout_length=1_basis=0.npy")
# p5 = np.load("mpc_rollout_length=5_basis=0.npy")
# p10 = np.load("mpc_rollout_length=10_basis=0.npy")
# p25 = np.load("mpc_rollout_length=25_basis=0.npy")
# p50 = np.load("mpc_rollout_length=50_basis=0.npy")

y1 = y[:, 0]

y5 = y[:, 0]
h = np.roll(y5, -1)
for i in range(5 - 1):
    h[-1] = 0
    y5 = np.append(y5, h)
    h = np.roll(h, -1)

y10 = y[:, 0]
h = np.roll(y10, -1)
for i in range(10 - 1):
    h[-1] = 0
    y10 = np.vstack((y10, h))
    h = np.roll(h, -1)
y10.reshape(len(y),10)

y25 = y[:, 0]
h = np.roll(y25, -1)
for i in range(25 - 1):
    h[-1] = 0
    y25 = np.vstack((y25, h))
    h = np.roll(h, -1)
y25.reshape(len(y),25)

y50 = y[:, 0]
h = np.roll(y50, -1)
for i in range(50 - 1):
    h[-1] = 0
    y50 = np.vstack((y50, h))
    h = np.roll(h, -1)
y50.reshape(len(y),50)