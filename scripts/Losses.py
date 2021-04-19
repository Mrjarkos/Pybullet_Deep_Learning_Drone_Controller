import numpy as np
import matplotlib.pyplot as plt

def MAE(error):
    n = len(error)
    return 1/n*sum(abs(error))

def MSE(error):
    n = len(error)
    return 1/n*sum(error**2)

def Huber(error, delta=2):
    n = len(error)
    return 1/n*(delta**2)*sum(np.sqrt(1+(error/delta)**2)-1)

# error =  np.arange(-1, 1, 0.01)

# hub = []
# for e in error:
#     hub.append(Huber(np.array([e]), delta=0.1))
# plt.plot(error, hub)

# hub = []
# for e in error:
#     hub.append(Huber(np.array([e]), delta=500))
# plt.plot(error, hub)

# mse = []
# for e in error:
#     mse.append(MSE(np.array([e])))
# plt.plot(error, mse)
# mae = []
# for e in error:
#     mae.append(MAE(np.array([e])))
# plt.plot(error, mae)
# plt.show()