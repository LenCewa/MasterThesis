import matplotlib.pyplot as plt
import numpy as np

lol = np.load("layers_1_17_17_1_iterations=500.npy")
plt.figure()
plt.plot(lol)
plt.show()
