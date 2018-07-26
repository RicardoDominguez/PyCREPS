import numpy as np
import pdb
import matplotlib.pyplot as plt
from simulator import Scenario
def predictReward(mod, x0, M, H, hipol):
    '''
    Compute expected rewards by sampling trayectories using the forward models.
    '''
    W = hipol.sample(M).T # (M x W)
    R = np.zeros(M).reshape(-1, 1)
    F = np.zeros((M, len(x0)))

    x0s = np.empty((M, 2))
    x0s[:, 0] = 200
    x0s[:, 1] = np.random.rand() * 0.5236 + 0.5236
    #pdb.set_trace()
    R = mod.simulateRobot(M, H, x0s, W.T)

    #plt.plot(R)
    #plt.show()
    #pdb.set_trace()
    return R, W, F
