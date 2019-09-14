# https://www.stewartcalculus.com/data/CALCULUS%20Concepts%20and%20Contexts/upfiles/3c3-NonhomgenLinEqns_Stu.pdf
# Example 2: y'' + 4y = 0 has soloution c_1 * cos(2x) + c_2 * sin(2x) which our approach should learn

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def harmonic_oscillator(y, t, b, c):
    v1, v2 = y
    dydt = [v2, -b * v2 - c * v1]
    return dydt

b, c = 1, 5
y0 = [1, 0]
t = np.linspace(0, 20, num=500)

y = odeint(harmonic_oscillator, y0, t, args=(b, c))

plt.plot(t, y[:, 0], 'b')
plt.plot(t, y[:, 1], 'g')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()