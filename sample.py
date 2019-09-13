from scipy import signal
import jax.numpy as jnp

def get_sampled_trajectory(system):
    if system == 'pendulum':
        print("Return sampled trajectory from Dynamical_Systems/pendulum.py")
    elif system == 'linear_ode':
        print("Return sampled trajectory from Dynamical_Systems/linear_ode.py")
    elif system == 'weaklyNL':
        print("Return sampled trajectory from Dynamical_Systems/weakly_non_linear.py")
    else:
        print("System not existing.")

def square_wave(min, max, num):
    values = jnp.linspace(min, max, num=num)
    labels = [signal.square(x, 0.5) for x in values]
    labels = jnp.array(labels)
    return values, labels