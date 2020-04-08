from Fourier import dFourier, ddFourier
from evaluate_parameter_koopman import *
import matplotlib.pyplot as plt

# Implementiere 1-Step predictions in Koopman space

# Newtons Optimization Method
dN=N#5 #Cutoff
ddN=N#5 #Cutoff
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

print("HELLO1")

Y = koopman_mpc_prediction(K, 5, steps, dim, 0).reshape(1,-1) #koopman_prediction(K, x0, steps, dim, 0).reshape(1,-1) # Array of predictions
print("HELLO2")
for i in range(dim - 1):
    print("HELLO",i+3)
    Y = np.append(Y, koopman_mpc_prediction(K, 5, steps, dim, i + 1).reshape(1,-1), axis=0) #np.append(Y, koopman_prediction(K, x0, steps, dim, i + 1).reshape(1,-1), axis=0) #

Y = np.array(Y)
print("Shape of Y:", Y.shape)

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

# res, err = newton_optimization_method(Y[:,0].reshape(-1,1), 3, 5)
# g_x = []
# for c in loc:
#     set_fourier_coefficients(c, N)
#     g_x += [fourier.predict(fourier.coefficients, res[-1])]
# g_x = np.array(g_x)
# print("y = ", Y[:, 0].reshape(-1,1))
# print(" g(x) = ", g_x)
# print("res: ", res)
# print("err: ", err)
reconstruct = []
reconstruct_error = []
steps = 20
x0 = np.pi - 1e-2
alpha0 = 1
damping0 = 0.999

res, err = newton_optimization_method(Y[:,0].reshape(-1,1), x0, steps, alpha0, damping0)
reconstruct += [res[-1]]
reconstruct_error += [err[-1]]
for t in range(Y.shape[1]-1): #Y.shape[1] Y.shape[1]-1
    print("Iteration: ", t, " / 500")
    res, err = newton_optimization_method(Y[:,t+1].reshape(-1,1), reconstruct[t], steps, alpha0, damping0)
    reconstruct += [res[-1]]
    reconstruct_error += [err[-1]]
    # save reconstrcted trajectory
    np.save("Reconstructions/" + "mpc_rollout_basis=" + str(basis_vector), reconstruct)
    np.save("Reconstruction_Errors/" + "mpc_rollout_basis=" + str(basis_vector), reconstruct_error)
    print("saved trajectory")

t = np.linspace(0, 20, num=500)
# Plot result
fig, ax = plt.subplots()
ax.plot(t, trajectory, label='x(t)')
ax.plot(t, reconstruct, label='{argmin_t ||Ψ(x_t)-y_n||² : t = 1, ..., 20}')
ax.set(xlabel='time (s)', ylabel='θ (rad)', title='Recovering the original trajectory from the Koopman predictions')
ax.grid()
plt.legend()
fig.savefig("mpcRolloutBasis="+str(basis_vector)+".pdf")
plt.show()


###########
# for t in range(Y.shape[1]): #Y.shape[1]
#     print("Iteration: ", t, " / 500")
#     if t in range(50):
#         x0 = 3.1
#         print("x0 = ", x0)
#     elif t in range(50,80):
#         x0 = 3
#         print("x0 = ", x0)
#     elif t in range(80,90):
#         x0 = 2.8
#         print("x0 = ", x0)
#     elif t in range(90,105) :
#         x0 = 2.6
#         print("x0 = ", x0)
#     elif t in range(105,115) :
#         x0 = 2.35
#         print("x0 = ", x0)
#     elif t in range(115,125):
#         x0 = 2
#         print("x0 = ", x0)
#     elif t in range(125,150):
#         x0 = 1.5
#         print("x0 = ", x0)
#     elif t in range(150, 165):
#         x0 = 0.8
#         print("x0 = ", x0)
#     elif t in range(165, 180):
#         x0 = 0.4
#         print("x0 = ", x0)
#     elif t in range(180, 200):
#         x0 = 0.2
#         print("x0 = ", x0)
#     else:
#         x0 = 0.1
#         print("x0 = ", x0)
#     res, err = newton_optimization_method(Y[:,t].reshape(-1,1), x0, steps, alpha0, damping0)
#     reconstruct += [res[-1]]
#     reconstruct_error += [err[-1]]


'''for t in range(Y.shape[1]): #Y.shape[1]
    print("Iteration: ", t, " / 500")
    if t in range(100):
        x0 = 3
        print("x0 = " ,x0)
    elif t in range(100,120) :
        x0 = 2.5
        print("x0 = ", x0)
    elif t in range(120,140):
        x0 = 2
        print("x0 = ", x0)
    elif t in range(140,160):
        x0 = 1.5
        print("x0 = ", x0)
    elif t in range(160, 180):
        x0 = 1
        print("x0 = ", x0)
    elif t in range(180, 200):
        x0 = 0.5
        print("x0 = ", x0)
    else:
        x0 = 0
        print("x0 = ", x0)'''