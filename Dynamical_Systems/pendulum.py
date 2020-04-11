# https://de.wikipedia.org/wiki/Mathematisches_Pendel
# Python implementation: https://pythonmatplotlibtips.blogspot.com/2018/01/solve-animate-single-pendulum-odeint-artistanimation.html
# S.79 Bsp. 9.10: https://www.mathematik.tu-darmstadt.de/media/analysis/lehrmaterial_anapde/hallerd/DGLSkript1819.pdf
# BLush implements a dynamical system and samples from this her data (PendulumFn.m)

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def pendulum(y, t, b, c):
    theta, omega = y
    dydt = [omega, -b * omega - c * np.sin(theta)]
    return dydt

b, c = 2, 3.0
y0 = [np.pi - 1e-2, 0.0]
t = np.linspace(0, 20, num=500)

y = odeint(pendulum, y0, t, args=(b, c))

#plt.plot(t, y[:, 0], 'b')
# plt.plot(t, y[:, 1], 'g')
# plt.legend(loc='best')
# plt.xlabel('t')
# plt.grid()
# plt.show()