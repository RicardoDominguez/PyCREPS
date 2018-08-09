import numpy as np
import pdb
import matplotlib.pyplot as plt
from simulator import Scenario

def sampleContext(N):
    S = np.empty((N, 2))
    S[:, 0] =  np.random.rand(N) * 170 + 50
    S[:, 1] =  np.random.rand(N) * 0.87 + 0.5236
    return S

def predictReward(mod, x0, M, N, H, hipol):
    '''
    Compute expected rewards by sampling trayectories using the forward models.
    '''
    R = np.zeros(M).reshape(-1, 1)
    # F = np.zeros((M, len(x0)))
    F = sampleContext(M) # (M x S)
    W = hipol.contextSample(S = F, N = M).T # (M x W)

    # Ws = np.repeat(W, N, axis = 0)
    R = mod.simulateRobot(M, H, F, W.T)

    # Rs = mod.simulateRobot(M*N, H, x0s, Ws.T) # (M * N)
    # R = np.mean(np.split(Rs, M, axis = 0), axis = 1)

    #plt.plot(R)
    #plt.show()
    #pdb.set_trace()
    return R, W, F
