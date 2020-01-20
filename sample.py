from scipy import signal
import jax.numpy as jnp
from Dynamical_Systems import weakly_pendulum, pendulum

def get_sampled_trajectory(system):
    trajectory = []
    if system == 'pendulum':
        print("Return sampled trajectory from Dynamical_Systems/pendulum.py")
        #trajectory = jnp.array(pendulum.y.ravel())
    elif system == 'weakly_pendulum':
        print("Return sampled trajectory from Dynamical_Systems/weakly_pendulum.py")
        trajectory = jnp.array(weakly_pendulum.y.ravel())
    elif system == 'linear_ode':
        print("Return sampled trajectory from Dynamical_Systems/harmonic_oscillator.py")
    elif system == 'weaklyNL':
        print("Return sampled trajectory from Dynamical_Systems/weakly_non_linear.py")
    else:
        print("System not existing.")

    return trajectory

def square_wave(min, max, num):
    values = jnp.linspace(min, max, num=num)
    labels = [signal.square(x, 0.5) for x in values]
    labels = jnp.array(labels)
    return values, labels

def random_fourier(min, max, num):
    values = jnp.linspace(min, max, num=num)
    labels = [5 * jnp.sin(x) + 2 * jnp.cos(4 * x) + 7 for x in values]
    labels = jnp.array(labels)
    return values, labels

def trigeonmetric_product(min, max, num):
    values = jnp.linspace(min, max, num=num)
    labels = [jnp.sin(x) * jnp.square(jnp.cos(x)) for x in values]
    labels = jnp.array(labels)
    return values, labels
