import numpy as np
import jax.numpy as jnp
from jax import vmap
import matplotlib.pyplot as plt

omega = 1
t = jnp.linspace(0, 20, num=500)

loc = np.load("test_run_N=6_iterations=450_dim=10.npy")
k = 9
coef5 = loc[k]#np.load("test_run_N=15_iterations=5000_dim=12.npy")[5]
coef5 = [jnp.array(coef5[0])] + [jnp.array(coef5[i]) for i in range(1, coef5.shape[0] - 1)]

def predict(coefficients, x):
    periodic_sum = 0
    a_0 = coefficients[0]
    for n in range(len(coefficients) - 1):
        a_n = coefficients[n + 1][0]
        b_n = coefficients[n + 1][1]
        periodic_sum += a_n * jnp.cos((n + 1) * omega * x) + b_n * jnp.sin((n + 1) * omega * x)
    return a_0 / 2 + periodic_sum

def dpredict(coefficients, x):
    periodic_sum = 0
    #a_0 = coefficients[0]
    for n in range(5):
        a_n = coefficients[n+1][0]
        b_n = coefficients[n+1][1]
        periodic_sum += -a_n * (n + 1) * jnp.sin((n + 1) * omega * x) + b_n * (n + 1) * jnp.cos((n + 1) * omega * x)
    return periodic_sum

def ddpredict(coefficients, x):
    periodic_sum = 0
    #a_0 = coefficients[0]
    for n in range(5):
        a_n = coefficients[n+1][0]
        b_n = coefficients[n+1][1]
        periodic_sum += -a_n * (n + 1)**2 * jnp.cos((n + 1) * omega * x) - b_n * (n + 1)**2 * jnp.sin((n + 1) * omega * x)
    return periodic_sum

batched_predict = vmap(predict, in_axes=(None, 0))
dbatched_predict = vmap(dpredict, in_axes=(None, 0))
ddbatched_predict = vmap(ddpredict, in_axes=(None, 0))

# preds1 = batched_predict(coef1, t)[:, 0]
# preds2 = batched_predict(coef2, t)[:, 0]
# preds3 = batched_predict(coef3, t)[:, 0]
# preds4 = batched_predict(coef4, t)[:, 0]
preds5 = batched_predict(coef5, t)[:, 0]
dpreds5 = dbatched_predict(coef5, t)#[:, 0]
ddpreds5 = ddbatched_predict(coef5, t)#[:, 0]


fig, ax = plt.subplots()
ax.plot(preds5, label="Ψ_" + str(k + 1))
ax.plot(dpreds5, label="dΨ_" + str(k + 1))
ax.plot(ddpreds5, label="ddΨ_" + str(k + 1))
ax.set(xlabel='x', ylabel='y', title='Observable function Ψ_'+ str(k + 1))
ax.grid()
plt.legend()
fig.savefig("observable" + str(k + 1) + ".pdf")
plt.show()



# Plot result
# plt.figure()
# # plt.plot(preds1, label="p1")
# # plt.plot(preds2, label="p2")
# # plt.plot(preds3, label="p3")
# # plt.plot(preds4, label="p4")
# plt.plot(preds5, label="Ψ_" + str(k + 1))
# plt.plot(dpreds5, label="dΨ_" + str(k + 1))
# plt.plot(ddpreds5, label="ddΨ_" + str(k + 1))
# plt.legend()
# plt.show()


# coef1 = np.load("test_run_N=6_iterations=100.npy")[0]
# coef1 = [jnp.array(coef1[0])] + [jnp.array(coef1[i]) for i in range(1, coef1.shape[0] - 1)]
#
# coef2 = np.load("test_run_N=8_iterations=10.npy")[0]
# coef2 = [jnp.array(coef2[0])] + [jnp.array(coef2[i]) for i in range(1, coef2.shape[0] - 1)]
#
# coef3 = np.load("test_run_N=8_iterations=1000.npy")[0]
# coef3 = [jnp.array(coef3[0])] + [jnp.array(coef3[i]) for i in range(1, coef3.shape[0] - 1)]
#
# coef4 = np.load("test_run_N=10_iterations=1000_dim=8.npy")[0]
# coef4 = [jnp.array(coef4[0])] + [jnp.array(coef4[i]) for i in range(1, 11)]
#
# coef5 = np.load("test_run_N=15_iterations=5000_dim=12.npy")[0]
# coef5 = [jnp.array(coef5[0])] + [jnp.array(coef5[i]) for i in range(1, coef5.shape[0] - 1)]


# coef1 = np.load("test_run_N=15_iterations=5000_dim=4.npy")[0]
# coef1 = [jnp.array(coef1[0])] + [jnp.array(coef1[i]) for i in range(1, coef1.shape[0] - 1)]
#
# coef2 = np.load("test_run_N=15_iterations=5000_dim=4.npy")[1]
# coef2 = [jnp.array(coef2[0])] + [jnp.array(coef2[i]) for i in range(1, coef2.shape[0] - 1)]
#
# coef3 = np.load("test_run_N=15_iterations=5000_dim=4.npy")[2]
# coef3 = [jnp.array(coef3[0])] + [jnp.array(coef3[i]) for i in range(1, coef3.shape[0] - 1)]
#
# coef4 = np.load("test_run_N=15_iterations=5000_dim=4.npy")[3]
# coef4 = [jnp.array(coef4[0])] + [jnp.array(coef4[i]) for i in range(1, coef4.shape[0] - 1)]