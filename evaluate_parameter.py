import numpy as np

loc = np.load("Koopman_Coefficients/test_run_N=8_iterations=10.npy")
lops = np.load("Koopman_Predictions/test_run_N=8_iterations=10.npy")

print(loc)