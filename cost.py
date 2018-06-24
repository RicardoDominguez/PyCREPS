import numpy as np

class CostFcn:
    def __init__(self, K, iCost, target = 0):
        self.K = K
        self.iCost = iCost
        self.target = target
        #self.maxV = maxV
        #self.outboundR = outboundR

    def sample(self, y):
        '''
        Returns the cost of N rollouts.
        Cost = (target - measurements), scaled by a factor k.

        Inputs
            y   states sensed       (N x H)

        Outputs
            C   cost each rollout   (N x 1)
        '''
        err = np.sum(np.abs(y[:, self.iCost] - self.target), 1)
        C = np.exp(-err / self.K)
        #if self.maxV != None:
        #    C[y[:][iCost] < self.maxV * 0.95] = 1
        return C.reshape(-1, 1)
