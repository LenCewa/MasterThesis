from Fourier import Fourier
from Fourier_Koopman_eigenfunctions import FourierKoopmanEigenfunctions
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sample import *
import json
import time

start_time = time.time()

# Hyperparameter
omega = 1
T = (2 * jnp.pi) / omega
step_size = 0.001
N = 5
#dim_subspace = 15
iterations = 100


# Do computation
values, labels = trigeonmetric_product(-10, 10, 500)
fourier = Fourier(T, omega, step_size, N, iterations, values, labels)
coefficients = fourier.compute_coefficients()
preds = fourier.batched_predict(coefficients, values)[:, 0]
losses = fourier.loss_list

execution_time = (time.time() - start_time) / 60

# Plot result
print("Execution time in mimutes: ", execution_time)
plt.plot(preds, label="pred")
#plt.plot(losses, label="losses")

# Timestamp for saving Fourier coefficients, hyperparameter and plots

dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Save plot
plt.savefig("Plots/" + dt + "_N=" + str(N) + ".png")

# Save Fourier coefficients
np.save("Fourier_Coefficients/" + dt + "_N=" + str(N), coefficients)

# Save Fourier list of losses
np.save("Fourier_Loss/" + dt + "_N=" + str(N), losses)

# Save labels
np.save("Labels/" + dt + "_N=" + str(N), labels)

# Save preditctions
np.save("Predictions/" + dt + "_N=" + str(N), preds)

# Save values
np.save("Values/" + dt + "_N=" + str(N), values)

# Save hyperparameter
data = {}
data['hyperparameter'] = []
data['hyperparameter'].append({
    'omega': str(omega),
    'T': str(T),
    'N': str(N),
    'step_size': str(step_size),
    'iterations': str(iterations),
    'execution_time_minutes': execution_time
})
with open("Hyperparameter/" + dt + ".txt", 'w') as outfile:
    json.dump(data, outfile)

plt.show()
