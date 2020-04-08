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
              [0, 0, 0, 0, 0, 0, 0, 3 * np.power(dt, 5), 15 * np.power(dt, 2) + 15 * np.power(dt, 4) + np.power(dt, 6), -6 * dt - 5 * np.power(dt, 3) + 9 * np.power(dt, 5) + np.power(dt, 7), a, c],
              [0, 0, 0, 0, 0, 0, 0, np.power(dt, 6), -20 * np.power(dt, 3) - 6 * np.power(dt, 5), 15 * np.power(dt, 2) - 5 * np.power(dt, 4) - 5 * np.power(dt, 6), b, d]])


def euler_prediction(start_value, steps):
    xk = start_value
    euler_preds = [xk]
    for s in range(steps):
        xk = euler_preds[s]
        euler_preds += [xk - np.sin(xk)*dt]
    return euler_preds


def koopman_prediction(steps, basis_vector):
    koopman_preds = []
    pred = 0
    for s in range(steps):
        for k in range(12):
            pred += np.linalg.matrix_power(K, s)[:, basis_vector][k] * basis[k]
        koopman_preds += [pred]
        pred = 0
    return np.array(koopman_preds)


def compute_function_space_error(sin_fitted_trajectory, sin_euler_preds, koopman_preds, cutoff):
    sin_euler_error = np.power(sin_fitted_trajectory[:(cutoff + 1)] - sin_euler_preds, 2)
    koopman_error = np.power(sin_fitted_trajectory[:cutoff] - koopman_preds, 2)
    return sin_euler_error, koopman_error


def get_fitted_trajectory(shift):
    return np.repeat(trajectory, shift)

koopman_preds = koopman_prediction(steps, 0) #1
euler_preds = euler_prediction(x0, steps)
sin_euler_preds = np.sin(euler_preds) #* np.cos(euler_preds)
fitted_trajectory = get_fitted_trajectory(shift_trajectory)
sin_fitted_trajectory = np.sin(fitted_trajectory) #* np.cos(fitted_trajectory)
sin_euler_error, koopman_error = compute_function_space_error(sin_fitted_trajectory, sin_euler_preds, koopman_preds, steps)


t = np.linspace(0, 20, num=20000)
t2= t[:4208]
fig, ax = plt.subplots()
ax.plot(t, sin_fitted_trajectory, label='sin(x(t))')
ax.plot(t2, koopman_preds, label='[K^t]sin(x0)')
ax.set(xlabel='time (s)', ylabel='sin(θ)', title='Predicting the simple pendulum with a 12-dim basis')
ax.grid()
plt.legend()
fig.savefig("12dimself.png")
plt.show()

# Plot result
# def configure_plots():
#     import matplotlib
#     from distutils.spawn import find_executable
#     matplotlib.rcParams['font.family'] = 'serif'
#     matplotlib.rcParams['figure.figsize'] = [19, 19]
#     matplotlib.rcParams['legend.fontsize'] = 26
#     matplotlib.rcParams['axes.titlesize'] = 42
#     matplotlib.rcParams['axes.labelsize'] = 42
#     if find_executable("latex"):
#         matplotlib.rcParams['text.usetex'] = True
#         matplotlib.rcParams['text.latex.unicode'] = True
#
# configure_plots()
#
# fig, (ax1, ax2) = plt.subplots(2, 1, sharey=True)
# ax1.set_xlabel('steps')# / predicted steps: ' + str(steps), **font)
# ax1.set_ylabel('fitted trajectory')#, **font)
# ax1.plot(koopman_preds, label="Koopman prediction", linewidth=10)
# #ax1.plot(sin_euler_preds, label="sin(Euler)", linewidth=10)
# ax1.plot(sin_fitted_trajectory, label="embedded trajectory", linewidth=5)
# ax2.plot(fitted_trajectory, label="trajectory", linewidth=5)
# '''ax2 = ax1.twinx()
# ax2.set_ylabel('MSE')
# ax2.plot(sin_euler_error, label="sin(Euler) MSE")
# ax2.plot(koopman_error, label="Koopman MSE")'''
# fig.tight_layout()
# ax1.legend(fontsize='xx-large')
# plt.show()
