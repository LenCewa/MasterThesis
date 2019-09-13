# The Lotka–Volterra equations, also known as the predator–prey equations, are a pair of first-order nonlinear differential equations.
# For more information cf: https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations and https://www.mathematik.tu-darmstadt.de/media/analysis/lehrmaterial_anapde/hallerd/DGLSkript1819.pdf

# Example found here: https://www.danham.me/r/2015/10/29/differential-eq.html

import matplotlib.animation as animation
from scipy.integrate import odeint
from pylab import *

def BoatFishSystem(state, t):
    fish, boat = state
    d_fish = fish * (2 - boat - fish)
    d_boat = -boat * (1 - 1.5 * fish)
    return [d_fish, d_boat]

t = arange(0, 20, 0.1)
init_state = [1, 1]
state = odeint(BoatFishSystem, init_state, t)

fig = figure()
xlabel('number of fish')
ylabel('number of boats')
plot(state[:, 0], state[:, 1], 'b-', alpha=0.2)

def animate(i):
    plot(state[0:i, 0], state[0:i, 1], 'b-')

ani = animation.FuncAnimation(fig, animate, interval=1)
show()