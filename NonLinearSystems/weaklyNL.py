# Example from my proposal (cf. https://arxiv.org/pdf/1712.09707.pdf on pg. 5)

import matplotlib.animation as animation
from scipy.integrate import odeint
from numpy import arange
from pylab import *

def weaklyNonLinear(state, t):
    x1, x2 = state
    d_x1 = -0.05*x1
    d_x2 = -1 *(x2 - x1**2)
    return [d_x1, d_x2]

t = arange(0, 20, 0.1)
init_state = [1, 1]
state = odeint(weaklyNonLinear, init_state, t)

fig = figure()
xlabel('x1')
ylabel('x2')
plot(state[:, 0], state[:, 1], 'b-', alpha=0.2)

def animate(i):
    plot(state[0:i, 0], state[0:i, 1], 'b-')

ani = animation.FuncAnimation(fig, animate, interval=1)
show()