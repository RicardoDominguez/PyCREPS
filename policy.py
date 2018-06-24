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
            nX         number of tiles
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
        u[ilX] = W[0, ilX]
        u[imX] = W[-1, imX]

        # For values whitin range...
        aoX = ~(ilX | imX)
        indxs = ceil(X(aoX) / float(self.deltaX)) # Indexes of W used
        u[aoX] = np.diag(W[indxs, aoX]) # Pairs of is and aoX (TODO: prolly wrong)

        # Check for saturation
        u[u > self.max] = self.max
        u[u < self.min] = self.min
