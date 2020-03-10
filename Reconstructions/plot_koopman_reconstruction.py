import matplotlib.pyplot as plt
import numpy as np


r = np.load("second_try_max_iter=10_tol=1e-3.npy")
#re = np.load("Reconstruction_Errors/second_try_max_iter=10_tol=1e-3.npy")

plt.figure()
plt.plot(r)
#plt.plot(re)
plt.show()