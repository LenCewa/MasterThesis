import numpy as np
from Dynamical_Systems import weakly_pendulum

time = 100
dt = 0.01
steps = weakly_pendulum.t[time] / dt
# TODO:
#  - 400 Schritte können wir sehr gut linear vorhersagen. Prüfe was passiert wenn du
#  - Prüfe was passiert wenn du ab dem 400sten Schritt wieder neu vorhersagst
trajectory = weakly_pendulum.y.ravel()
x0 = weakly_pendulum.y0
dim = 4

def set_basis(dim):
    basis = []
    for k in range(dim):
        basis += [np.sin(x0) * np.power(np.cos(x0), k)]
    return basis

def compute_K(dim, dt, power):
    row0 = np.zeros((1, dim))
    row0[0][0] = 1
    row1 = np.zeros((1, dim))
    row1[0][0] = dt
    row1[0][1] = 1
    row = np.append(row0, row1, axis=0)
    for s in range(2, dim):
        row = np.append(row, np.roll(row[s - 1, :].reshape(1, -1), 1), axis=0)
    row[-2][-1] = dt
    K = np.linalg.matrix_power(row, power)
    return row, K

row, K = compute_K(dim, dt, int(steps))
basis = set_basis(dim)

def Euler_approx():
    xk = x0
    for _ in range(int(steps)):
        xk = xk + np.sin(xk)*dt
    return xk

pred = 0
for k in range(dim):
    pred += K[:, 0][k] * basis[k]

print(weakly_pendulum.t[time], trajectory[time])
print("Prediction: ", pred)
print("sin(label): ", np.sin(trajectory[time]), " label: ", trajectory[time])
print("sin(Euler): ", np.sin(Euler_approx()), " Euler: ", Euler_approx())
print((np.sin(trajectory[time]) - pred)**2)