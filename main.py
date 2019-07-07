import gym
import quanser_robots
env = gym.make('Pendulum-v0')
env.reset()
env.render()
