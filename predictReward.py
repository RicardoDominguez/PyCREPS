import numpy as np
import pdb
import matplotlib.pyplot as plt

def predictReward(scn, x0, M, H, hipol, pol, cost):
    '''
    Compute expected rewards by sampling trayectories using the forward models.
    '''
    W = hipol.sample(M).T # (M x W)
    R = np.zeros(M).reshape(-1, 1)
    F = np.zeros((M, len(x0)))

    for e in range(M): # For each episode
        if np.remainder(e, 10) == 0: print('Simulation ' + str(e+1) + '...')
        w = W[e, :]
        scn.initScenario(x0)
        x = x0
        for t in xrange(H): # For each step within horizon
            u = pol.sample(w, x)
            y = scn.step(u)
            #print u
            #pdb.set_trace()
            R[e] += cost.sample(y).reshape(-1,)
            x = y
            #scn.plot()

    #plt.plot(R)
    #plt.show()
    #pdb.set_trace()
    return R, W, F
