import matplotlib.pyplot as plt
import numpy as np

from Dynamical_Systems import weakly_pendulum

# trajectory of the weakly pendulum
y = weakly_pendulum.y

p1 = np.load("mpc_rollout_length=1_basis=0.npy")
p5 = np.load("mpc_rollout_length=5_basis=0.npy")
p10 = np.load("mpc_rollout_length=10_basis=0.npy")
p25 = np.load("mpc_rollout_length=25_basis=0.npy")
p50 = np.load("mpc_rollout_length=50_basis=0.npy")

y1 = y[:, 0]
rmse1 = np.sqrt(np.sum((p1-y1)**2)/len(p1))
print("RMSE 1 = ", rmse1)

y5 = y[:, 0]
h = np.roll(y5, -1)
for i in range(5 - 1):
    h[-1] = 0
    y5 = np.append(y5, h)
    h = np.roll(h, -1)
rmse5 = np.sqrt(np.sum((p5-y5)**2)/len(p5))
print("RMSE 5 = ", rmse5)

y10 = y[:, 0]
h = np.roll(y10, -1)
for i in range(10 - 1):
    h[-1] = 0
    y10 = np.append(y10, h)
    h = np.roll(h, -1)
rmse10 = np.sqrt(np.sum((p10-y10)**2)/len(p10))
print("RMSE 10 = ", rmse10)

l25 = int(len(p25) / 25)
y25 = y[:l25, 0]
h = np.roll(y25, -1)
for i in range(25 - 1):
    y25 = np.append(y25, h)
    h = np.roll(h, -1)
rmse25 = np.sqrt(np.sum((p25-y25)**2)/len(p25))
print("RMSE 25 = ", rmse25)


l50 = int(len(p50) / 50)
y50 = y[:l50, 0]
h = np.roll(y50, -1)
for i in range(50 - 1):
    y50 = np.append(y50, h)
    h = np.roll(h, -1)
rmse50 = np.sqrt(np.sum((p50-y50)**2)/len(p50))
print("RMSE 50 = ", rmse50)