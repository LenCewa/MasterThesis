# Implement y' = sin(y)
# Check if spectrum is continous
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def weakly_pendulum(y, t):
    dydt = np.sin(y)
    return dydt

# Initial condition
y0 = 1e-2 #1e-4 dann geht auch time = 210 TODO: Prüfe np.pi als Startwert (müsste das selbe wie 0 ergeben)

# Time points
t = np.linspace(0, 20, num=500)

# Solve ODE
y = odeint(weakly_pendulum, y0, t)

'''
# Plot result
plt.plot(t, y)
plt.xlabel('time')
plt.ylabel('y(t)')
plt.legend()
plt.show()
'''