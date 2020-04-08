from Fourier import dFourier, ddFourier
from evaluate_parameter_koopman import *
import matplotlib.pyplot as plt

# Newtons Optimization Method
dN=N
ddN=N
d_fourier = dFourier(1, 1, 0.001, N, 450, [], [])
dd_fourier = ddFourier(1, 1, 0.001, N, 450, [], [])

def set_dfourier_coefficients(c, N):
    for i in range(N + 1):
        # N = 15 impliziert [a0, [a1, b1], ..., [a15, b15]] also 16 Listenelemente
        d_fourier.coefficients[i] = jnp.array(c[i])

def set_ddfourier_coefficients(c, N):
    for i in range(N + 1):
        # N = 15 impliziert [a0, [a1, b1], ..., [a15, b15]] also 16 Listenelemente
        dd_fourier.coefficients[i] = jnp.array(c[i])

def Loss(x, y):
    gx = []
    for c in loc:
        set_fourier_coefficients(c, N)
        gx += [fourier.predict(fourier.coefficients, x)]
    gx = np.array(gx)
    return np.linalg.norm(gx - y)**2

def dLoss(x, y):
    gx = []
    dgx = []
    dL = 0
    for c in loc:
        set_fourier_coefficients(c, N)
        set_dfourier_coefficients(c, dN)
        gx += [fourier.predict(fourier.coefficients, x)]
        dgx += [d_fourier.predict(d_fourier.coefficients, x)]

    gx = np.array(gx)
    dgx = np.array(dgx)
    for i in range(dim):
        dL += 2 * (y[i] - gx[i]) * (-dgx[i])

    return dL

def ddLoss(x, y):
    gx = []
    dgx = []
    ddgx = []
    ddL = 0
    for c in loc:
        set_fourier_coefficients(c, N)
        set_dfourier_coefficients(c, dN)
        set_ddfourier_coefficients(c, ddN)
        gx += [fourier.predict(fourier.coefficients, x)]
        dgx += [d_fourier.predict(d_fourier.coefficients, x)]
        ddgx += [dd_fourier.predict(dd_fourier.coefficients, x)]

    gx = np.array(gx)
    dgx = np.array(dgx)
    ddgx = np.array(ddgx)

    for i in range(dim):
        ddL += 2 * (dgx[i] ** 2 - (y[i] - gx[i]) * ddgx[i])

    return ddL


def Loss_alphad(x, y, alpha, d):
    gx_alphad = []
    for c in loc:
        set_fourier_coefficients(c, N)
        gx_alphad += [fourier.predict(fourier.coefficients, x + alpha * d)]
    gx_alphad = np.array(gx_alphad)
    return np.linalg.norm(gx_alphad - y)**2

def newton_optimization_method(y, x0, iterations, alpha0, damping0):
    res = [x0]
    err = []
    alpha = alpha0
    damping = damping0
    roh = [1.2, 0.5, 1, 0.5, 0.01]

    for k in range(iterations):
        x = res[k]
        L = Loss(x, y)

        err += [L]
        # print("Iteration: k = ", k, "--> Error ||y - g(x_k)||^2 = ", L, " for x = ", x)
        if L < 1e-10: break

        dL = dLoss(x, y)[0]
        ddL = ddLoss(x, y)[0]

        d = -dL / (ddL + damping)
        L_alphad = Loss_alphad(x, y, alpha, d)
        i = 0

        while L_alphad > (L + roh[4]*dL * alpha * d):
            # print("Iteration: ", k, " while-loop: ", i)
            # print("f(x + alpha * d) = ", gx_alphad, " > f(x) + r*f'(x) = ", g_x + roh[4] * dL)
            i += 1
            alpha = roh[1] * alpha
            # Optionally:
            damping = roh[2] * damping
            d = -dL / (ddL + damping)
            L_alphad = Loss_alphad(x, y, alpha, d)

        x = x + alpha * d
        res += [x]
        alpha = np.min([roh[0], alpha, 1])

        # Optinally:
        damping = roh[3] * damping

    return res, err


reconstruct = []
reconstruct_error = []
alpha0 = 1
damping0 = 0.999

rollout_list = [10, 1]#[1, 5, 10, 25, 50]
for r in rollout_list:
    print("H1")
    Y = koopman_mpc_rollout_prediction(K, r, steps, dim, 0).reshape(1, -1)
    print(Y.shape)
    for i in range(dim - 1):
        print("Compute Y, iteration = ",i)
        Y = np.append(Y, koopman_mpc_rollout_prediction(K, r, steps, dim, i + 1).reshape(1, -1), axis=0)

    Y = np.array(Y)
    print("Shape of Y:", Y.shape)
    reconstruct = []
    reconstruct_error = []
    newton_steps = 20
    for t in range(len(trajectory)): #Y.shape[1] Y.shape[1]-1
        print("Iteration: ", t, " / 500 and rollout = ", r)
        res, err = newton_optimization_method(Y[:, t*r].reshape(-1, 1), trajectory[t], newton_steps, alpha0, damping0)
        reconstruct += [res[-1]]
        reconstruct_error += [err[-1]]
        for m in range(r-1):
            res, err = newton_optimization_method(Y[:, t*r+m+1].reshape(-1, 1), reconstruct[t*r+m], newton_steps, alpha0, damping0)
            reconstruct += [res[-1]]
            reconstruct_error += [err[-1]]
        # save reconstrcted trajectory
        np.save("Reconstructions_RollOut/" + "mpc_rollout_length=" + str(r) + "_basis=" + str(basis_vector), reconstruct)
        np.save("Reconstruction_Errors_RollOut/" + "mpc_rollout_length=" + str(r) + "_basis=" + str(basis_vector), reconstruct_error)
        print("saved params")