def get_sampled_trajectory(system):
    if system == 'pendulum':
        print("Return sampled trajectory from DynamicalSystems/pendulum.py")
    elif system == 'linear_ode':
        print("Return sampled trajectory from DynamicalSystems/linear_ode.py")
    elif system == 'weaklyNL':
        print("Return sampled trajectory from DynamicalSystems/weakly_non_linear.py")
    else:
        print("System not existing.")