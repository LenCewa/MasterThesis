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
    return koopman_preds

def compute_error():
    # TODO: Compute error w.r.t. to different predictions
    # print((np.sin(trajectory[time]) - pred)**2)
    return -1

K = compute_operator(dim, dt)
koopman_preds = koopman_prediction(K, x0, steps, dim, 0)
euler_preds = euler_prediction(x0, steps)
sin_euler_preds = np.sin(euler_preds)
sin_trajectory = np.sin(trajectory)  # TODO: X-Achse anpassen


# Plot result
plt.figure()
plt.plot(koopman_preds, label="Koopman")
plt.plot(sin_euler_preds, label="sin(Euler)")
plt.plot(sin_trajectory, label="sin(trajectory)")
plt.legend()
plt.show()
