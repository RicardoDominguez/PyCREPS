# Implements the low level policy: (return an action given the system state)
import numpy as np

class TilePol:
    '''
    Policy in which the state spaced is tiled, with each tile having an
    associated index in the policy weight matrix denoting its control action
    '''
    def __init__(self, nX, deltaX, min, max):
        '''
        Inputs:
            nX         number of tiles = nW
            deltaX     spacing between each tile (tiles are even spaced)
            min        minimum control action
            max        maximum control action
        '''
        self.nX = nX
        self.deltaX = deltaX
        self.min = min
        self.max = max

    def sample(self, W, X):
        '''
        Inputs:
            W   policy weights                  (nW x T)
            X   vector of states                (T  x 1)
        '''
        T = X.shape[0]
        u = np.zeros((T, 1))

        # Check values out of tile range
        ilX, imX = X <= 0, X >= self.nX * self.deltaX
        u[ilX] = W[0, ilX.reshape(-1)]
        u[imX] = W[-1, imX.reshape(-1)]

        # For values whitin range...
        aoX = ~(ilX | imX).reshape(-1)
        indxs = int(np.ceil(X[aoX] / float(self.deltaX)))-1 # Indexes of W used
        u[aoX] = np.diag(W[indxs, aoX]) # Pairs of is and aoX

        # Check for saturation
        u[u > self.max] = self.max
        u[u < self.min] = self.min
        return u