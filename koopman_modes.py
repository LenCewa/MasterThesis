from Fourier import dFourier, ddFourier
from evaluate_parameter_koopman import *
import matplotlib.pyplot as plt

# Implementiere 1-Step predictions in Koopman space

# Newtons Optimization Method
dN=5 #Cutoff
ddN=5 #Cutoff
d_fourier = dFourier(1, 1, 0, dN, 0, [], [])
dd_fourier = ddFourier(1, 1, 0, ddN, 0, [], [])

def set_dfourier_coefficients(c, N):
    for i in range(N + 1):
        # N = 15 impliziert [a0, [a1, b1], ..., [a15, b15]] also 16 Listenelemente
        d_fourier.coefficients[i] = jnp.array(c[i])

def set_ddfourier_coefficients(c, N):
    for i in range(N + 1):
        # N = 15 impliziert [a0, [a1, b1], ..., [a15, b15]] also 16 Listenelemente
        dd_fourier.coefficients[i] = jnp.array(c[i])

Y = koopman_prediction(K, x0, steps, dim, 0).reshape(1,-1) # Array of predictions
for i in range(dim - 1):
    Y = np.append(Y, koopman_prediction(K, x0, steps, dim, i + 1).reshape(1,-1), axis=0)

Y = np.array(Y)
print("Shape of Y:", Y.shape)

def newton_optimization_method(y, x0, iterations):
    res = [x0]
    err = []

    for k in range(iterations):
        g_x, dg_x, ddg_x = [], [], []

        for c in loc:
            set_fourier_coefficients(c, N)
            set_dfourier_coefficients(c, dN)
            set_ddfourier_coefficients(c, ddN)
            g_x += [fourier.predict(fourier.coefficients, res[k])]
            dg_x += [d_fourier.predict(d_fourier.coefficients, res[k])]
            ddg_x += [dd_fourier.predict(dd_fourier.coefficients, res[k])]

        g_x, dg_x, ddg_x = np.array(g_x), np.array(dg_x), np.array(ddg_x)
        #print("SHAPE g(x) = ", g_x.shape)
        #print("g(x) = ", g_x)
        err += [np.linalg.norm(y - g_x)]
        print("Error ||y - g(x_k)||^2 = ", err[k])
        if err[k] < 1e-3: break
        dL, ddL = 0, 0
        for i in range(dim):
            dL += 2 * (y[i] - g_x[i]) * (-dg_x[i])
            ddL += 2 * (dg_x[i]**2 - (y[i] - g_x[i]) * ddg_x[i])

        print("dL = ", dL)
        print("ddL = ", ddL)
        x = res[k] - dL[0] / ddL[0]
        res += [x]
        print("intermediate approximations res = ", res, " in iterartion: ", k)

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
x0 = 3
for t in range(Y.shape[1]): #Y.shape[1]
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
        print("x0 = ", x0)

    res, err = newton_optimization_method(Y[:,0].reshape(-1,1), x0, 10)
    reconstruct += [res[-1]]
    reconstruct_error += [err[-1]]

# save reconstrcted trajectory
np.save("Reconstructions/" + "second_try_max_iter=10_tol=1e-3", reconstruct)
np.save("Reconstruction_Errors/" + "second_try_max_iter=10_tol=1e-3", reconstruct_error)
# Plot result
plt.figure()
plt.plot(reconstruct, "r.-",   label="reconstruction")
plt.plot(trajectory, "b.-", label="true trajectory")
plt.plot(reconstruct_error, "k.-", label="reconstruction error")
plt.ylim(-1,4)
plt.legend()
plt.show()
