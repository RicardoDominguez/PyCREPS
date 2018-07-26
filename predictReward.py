import numpy as np
import pdb
import matplotlib.pyplot as plt
from simulator import Scenario
def predictReward(mod, x0, M, N, H, hipol):
    '''
    Compute expected rewards by sampling trayectories using the forward models.
    '''
    W = hipol.sample(M).T # (M x W)
    R = np.zeros(M).reshape(-1, 1)
    F = np.zeros((M, len(x0)))

    Ws = np.repeat(W, N, axis = 0)

    x0s = np.empty((M*N, 2))
    x0s[:, 0] = 200
    x0s[:, 1] = np.random.rand(M*N) * 0.87 + 0.5236
    #pdb.set_trace()
    Rs = mod.simulateRobot(M*N, H, x0s, Ws.T) # (M * N)
    R = np.mean(np.split(Rs, M, axis = 0), axis = 1)

    #plt.plot(R)
    #plt.show()
    #pdb.set_trace()
    return R, W, F
