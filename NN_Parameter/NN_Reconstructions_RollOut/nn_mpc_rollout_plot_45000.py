import matplotlib.pyplot as plt
import numpy as np

from Dynamical_Systems import weakly_pendulum

# trajectory of the weakly pendulum
y = weakly_pendulum.y

# p1 = np.load("mpc_rollout_length=1_basis=0.npy")
p5 = np.load("mpc_rollout_length=5.npy")
p10 = np.load("mpc_rollout_length=10.npy")
p25 = np.load("mpc_rollout_length=25.npy")
p50 = np.load("mpc_rollout_length=50.npy")

p5 = p5.reshape(len(y), 5)
t5 = []
for k in range(len(y)):
    t5 += [np.arange(k, k + 5)]
t5 = np.array(t5)

p10 = p10.reshape(len(y), 10)
t10 = []
for k in range(len(y)):
    t10 += [np.arange(k, k + 10)]
t10 = np.array(t10)

l25 = int(len(p25) / 25)
p25 = p25.reshape(l25, 25)
t25 = []
for k in range(l25):
    t25 += [np.arange(k, k + 25)]
t25 = np.array(t25)

l50 = int(len(p50) / 50)
p50 = p50.reshape(l50, 50)
t50 = []
for k in range(l50):
    t50 += [np.arange(k, k + 50)]
t50 = np.array(t50)

mpc5 = zip(t5, p5)
mpc10 = zip(t10, p10)
mpc25 = zip(t25, p25)
mpc50 = zip(t50, p50)

fig, ax = plt.subplots()

for a, b in mpc50:
    ax.plot(a, b, 'y', label='mpc50')

for a, b in mpc25:
    ax.plot(a, b, 'c', label='mpc25')

for a, b in mpc10:
    ax.plot(a, b, 'g', label='mpc10')

for a, b in mpc5:
    ax.plot(a, b, 'r', label='mpc5')

ax.plot(y, label='x(t)')
ax.set(xlabel='time-steps', ylabel='θ (rad)', title='Comparing different prediction horizons')
ax.grid()
# plt.legend()
fig.savefig("nn_mpc_45000.pdf")
plt.show()
plt.figure()
