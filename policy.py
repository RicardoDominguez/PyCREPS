# Implements the low level policy: (return an action given the system state)
import numpy as np

class TilePol:
    '''
    Policy in which the state spaced is tiled, with each tile having an
    associated index in the policy weight matrix denoting its control action.
    '''
    def __init__(self, nX, deltaX, min, max, startX = 0):
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
        self.startX = startX

    def sample(self, W, X):
        '''
        Inputs:
            W   policy weights                  (nW x T)
            X   vector of states                (T  x 1)
        '''
        if isinstance(X, np.ndarray): # Many inputs
            T = X.shape[0]
            u = np.zeros((T))

            # Check values out of tile range
            ilX = X < self.startX
            imX = X > self.startX + self.deltaX * (self.nX - 2)
            u[ilX.reshape(-1)] = W[0, ilX.reshape(-1)]
            u[imX.reshape(-1)] = W[-1, imX.reshape(-1)]

            # For values whitin range...
            aoX = ~(ilX | imX).reshape(-1)
            if any(aoX):
                indxs = (np.floor((X[aoX] - self.startX) / float(self.deltaX))).astype(int) + 1# Indexes of W used
                u[aoX] = np.diag(W[indxs, aoX]) # Pairs of is and aoX

            # Check for saturation
            u[u > self.max] = self.max
            u[u < self.min] = self.min

        else: # Single input
            # Values out of tile range
            if X > self.startX + self.deltaX * (self.nX - 2):
                u = W[-1]
            elif X < 0:
                u = W[0]

            # Values within tile range
            else:
                indx = int((X - self.startX) / float(self.deltaX)) + 1 # Indexes of W used
                u = W[indx]

            # Saturation
            if u > self.max:
                u = self.max
            elif u < self.min:
                u = self.min

        return u

# if __name__ == "__main__":
#     pol = TilePol(5, 1, -2, 10, -2)
#     ws = np.array([[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5]])
#     print pol.sample(ws , np.array([-3, -1, 2]).reshape(-1,1))
