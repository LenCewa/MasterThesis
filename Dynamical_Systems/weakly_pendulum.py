# Implement y' = sin(y)
# Check if spectrum is continous
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def weakly_pendulum(y, t):
    dydt = -np.sin(y)
    return dydt

# Initial condition
y0 = np.pi - 1e-2 #1e-4 dann geht auch time = 210

# Time points
t = np.linspace(0, 20, num=500)

# Solve ODE
y = odeint(weakly_pendulum, y0, t)
# z = np.concatenate((y, y), axis=1)
# np.savetxt("simple_pendulum_2.csv", z, delimiter=",")
# print(y[:10])
# Plot result
# fig, ax = plt.subplots()
# ax.plot(y, label='x(t)')
# #ax.plot(t, np.ones(len(t)) * (np.pi - 1))
# ax.set(xlabel='time-steps', ylabel='θ (rad)', title='Trajectory of the simple pendulum')
# ax.grid()
# plt.legend()#plt.legend()
# fig.savefig("simpen.pdf")
# plt.show()