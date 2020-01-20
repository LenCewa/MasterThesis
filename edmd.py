from sample import *
import numpy as np
import matplotlib.pyplot as plt

# Berechne EDMD mit unseren Basisfunktionen und vergleiche Approximation


trajectory = get_sampled_trajectory("weakly_pendulum")
X = trajectory[:-1]
Y = trajectory[1:]
P = len(trajectory)
dim = 12

def basis(x):
    basis = [np.sin(x), np.sin(x) * np.cos(x), np.sin(x) * np.power(np.cos(x), 2), np.power(np.sin(x), 3),
             np.power(np.sin(x), 3) * np.cos(x), np.power(np.cos(x), 3) * np.sin(x),
             np.power(np.sin(x), 3) * np.power(np.cos(x), 2), np.power(np.sin(x), 3) * np.power(np.cos(x), 3),
             np.power(np.sin(x), 6), np.power(np.sin(x), 6) * np.cos(x),
             np.power(np.sin(x), 6) * np.power(np.cos(x), 2), np.power(np.sin(x), 6) * np.power(np.cos(x), 3)]
    return basis

def A_matrix(gX, gY):
    A = np.zeros((dim, dim))
    for p in range(gX.shape[0]):
        A += np.matmul(gX[p].reshape(-1,1), gY[p].reshape(1,-1))
    return A

def G_matrix(gX):
    G = np.zeros((dim, dim))
    for p in range(gX.shape[0]):
        G += np.matmul(gX[p].reshape(-1,1), gX[p].reshape(1,-1))
    return G

def koopman_operator():
    gX, gY= [], []
    for x in X:
        gX += [basis(x)]
    for y in Y:
        gY += [basis(y)]

    gX, gY = np.array(gX), np.array(gY)
    
    A = 1/P * A_matrix(gX, gY)
    G = 1/P * G_matrix(gX)
    
    G = np.linalg.inv(G)
    K = np.matmul(G, A)

    return K


K = koopman_operator()

print("Shape of the Koopman operator: ", K.shape)
print(K)

basis_vector = 0
x0 = np.pi - 1e-2
steps = 100
pred = 0

koopman_preds = []
basis = basis(x0)
for s in range(steps):
    for k in range(dim):
        pred += np.linalg.matrix_power(K, s)[:, basis_vector][k] * basis[k]
    koopman_preds += [pred]
    pred = 0


plt.plot(koopman_preds, label='koopman preds')
plt.plot(np.sin(trajectory), label='lifted trajectory')
plt.legend()
plt.show()