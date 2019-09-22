import numpy as np
from Dynamical_Systems import weakly_pendulum
import matplotlib.pyplot as plt

# Hyperparameter
dim = 4  # Does not give any major improvements when I increase the dimension (even by a factor of 1e3
time_index = 105  # If start value y0 = 1e-4 one can choose 210 for the time index
time = weakly_pendulum.t[time_index]
dt = 0.001
steps = int(time / dt)
trajectory = weakly_pendulum.y.ravel()
x0 = weakly_pendulum.y0
shift_trajectory = int(weakly_pendulum.t[1] / dt)


def set_basis(dim, start_value):
    basis = []
    for k in range(dim):
        basis += [np.sin(start_value) * np.power(np.cos(start_value), k)]
    return basis


def compute_operator(dim_subspace, dt):
    row0 = np.zeros((1, dim_subspace))
    row0[0][0] = 1
    row1 = np.zeros((1, dim_subspace))
    row1[0][0] = dt
    row1[0][1] = 1
    row = np.append(row0, row1, axis=0)
    for s in range(2, dim_subspace):
        row = np.append(row, np.roll(row[s - 1, :].reshape(1, -1), 1), axis=0)
    row[-2][-1] = dt
    return row


def euler_prediction(start_value, steps):
    xk = start_value
    euler_preds = [xk]
    for s in range(steps):
        xk = euler_preds[s]
        euler_preds += [xk + np.sin(xk)*dt]
    return euler_preds


def koopman_prediction(K, start_value, steps, dim_subspace, basis_vector):
    '''
    Computes prediction sin function space
    :param K:
    :param start_value:
    :param steps:
    :param dim_subspace:
    :param basis_vector: 0 = sin, 1 = sin*cos, 2 = sin * cos², 3 = sin * cos⁴, 4 = ...
    :return:
    '''
    basis = set_basis(dim_subspace, start_value)
    koopman_preds = []
    pred = 0
    for s in range(steps):
        for k in range(dim):
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


K = compute_operator(dim, dt)
koopman_preds = koopman_prediction(K, x0, steps, dim, 0)
euler_preds = euler_prediction(x0, steps)
sin_euler_preds = np.sin(euler_preds)
fitted_trajectory = get_fitted_trajectory(shift_trajectory)
sin_fitted_trajectory = np.sin(fitted_trajectory)
sin_euler_error, koopman_error = compute_function_space_error(sin_fitted_trajectory, sin_euler_preds, koopman_preds, steps)


# Plot result
fig, ax1 = plt.subplots()
ax1.set_xlabel('steps / predicted steps: ' + str(steps))
ax1.set_ylabel('trajectory')
ax1.plot(koopman_preds, label="Koopman")
ax1.plot(sin_euler_preds, label="sin(Euler)")
ax1.plot(sin_fitted_trajectory, label="sin(trajectory)")
ax1.plot(fitted_trajectory, label="trajectory")
ax2 = ax1.twinx()
ax2.set_ylabel('MSE')
ax2.plot(sin_euler_error, label="sin(Euler) MSE")
ax2.plot(koopman_error, label="Koopman MSE")
fig.tight_layout()
fig.legend()
plt.show()
