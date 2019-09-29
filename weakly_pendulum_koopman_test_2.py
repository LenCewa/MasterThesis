import numpy as np
from Dynamical_Systems import weakly_pendulum
import matplotlib.pyplot as plt

# Hyperparameter
time_index = 105  # If start value y0 = 1e-4 one can choose 210 for the time index
time = weakly_pendulum.t[time_index]
dt = 0.001
steps = int(time / dt)
trajectory = weakly_pendulum.y.ravel()
x0 = weakly_pendulum.y0
shift_trajectory = int(weakly_pendulum.t[1] / dt)
basis = [np.sin(x0), np.sin(x0) * np.cos(x0), np.sin(x0) * np.power(np.cos(x0), 2), np.power(np.sin(x0), 3),
         np.power(np.sin(x0), 3) * np.cos(x0), np.power(np.cos(x0), 3) * np.sin(x0),
         np.power(np.sin(x0), 3) * np.power(np.cos(x0), 2), np.power(np.sin(x0), 3) * np.power(np.cos(x0), 3),
         np.power(np.sin(x0), 6), np.power(np.sin(x0), 6) * np.cos(x0),
         np.power(np.sin(x0), 6) * np.power(np.cos(x0), 2), np.power(np.sin(x0), 6) * np.power(np.cos(x0), 3)]

a = 1 + 3 * np.power(dt, 2) - 10 * np.power(dt, 4) + 4 * np.power(dt, 6) + np.power(dt, 8)
b = -6 * dt + 10 * np.power(dt, 3) - 4 * np.power(dt, 5) - 4 * np.power(dt, 7)
c = -3 * dt + 27 * np.power(dt, 3) - 21 * np.power(dt, 5) + 45 * np.power(dt, 6) - 15 * np.power(dt, 7)
d = 1 - 3 * np.power(dt, 2) - 20 * np.power(dt, 3) - 17 * np.power(dt, 6) + 3 * np.power(dt, 8)

K = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [-dt, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, -dt, 1, 0, 0, -dt, 0, 0, 0, 0, 0, 0],
              [0, dt, np.power(dt, 2), 1, dt, 0, np.power(dt, 2), 0, 0, 0, 0, np.power(dt, 3)],
              [0, -np.power(dt, 2), 2 * dt - np.power(dt, 3), -3 * dt, 1 - 3 * np.power(dt, 2), 3 * np.power(dt, 2), -3 * np.power(dt, 3), 3 * np.power(dt, 2), 0, 0, 0, -6 * np.power(dt, 4)],
              [0, 0, -dt, 0, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, -2 * dt, 3 * np.power(dt, 2), -3 * dt + 4 * np.power(dt, 3), 3 * dt - 3 * np.power(dt, 3), 1 - 3 * np.power(dt, 2) + 4 * np.power(dt, 4), np.power(dt, 3) + 3 * np.power(dt, 5), 0, 0, 0, 0],
              [0, 0, 0, np.power(dt, 3), 3 * np.power(dt, 2) + np.power(dt, 4), -3 * np.power(dt, 2), 3 * dt + 7 * np.power(dt, 3) + np.power(dt, 5), 1 - 6 * np.power(dt, 2) + 12 * np.power(dt, 4), 0, 0, 0, 0],
              [0, 0, 0, 0, 0, np.power(dt, 3), 0, np.power(dt, 3), 1, -dt, np.power(dt, 2), 0],
              [0, 0, 0, 0, 0, -np.power(dt, 4), 0, -3 * np.power(dt, 4), -6 * dt, 1 - 6 * np.power(dt, 2), 2 * dt - 6 * np.power(dt, 3), 3 * np.power(dt, 2)],
              [0, 0, 0, 0, 0, 0, 0, 3 * np.power(dt, 5), 15 * np.power(dt, 2) + 15 * np.power(dt, 4) + np.power(dt, 6), -6 * dt - 5 * np.power(dt, 3) + 9 * np.power(dt, 5), np.power(dt, 7), a, c],
              [0, 0, 0, 0, 0, 0, 0, np.power(dt, 6), -20 * np.power(dt, 3) - 6 * np.power(dt, 5), 15 * np.power(dt, 2) - 5 * np.power(dt, 4) - 5 * np.power(dt, 6), b, d]])

print(len(basis))
print(K.shape)
print(K)

