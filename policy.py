# Implements the low level policy: (return an action given the system state)
import numpy as np
import pdb

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

class Proportional:
    def __init__(self, min, max, target, offset):
        self.min = min
        self.max = max
        self.target = target # (2, )
        self.offset = offset

    def sample(self, W, X):
        '''
        Inputs:
            W   policy weights                  (4 x 1)
            X   vector of states                (2, )
        '''
        e = (self.target - X).reshape(-1, 1) # error
        if(e[0] >= 0):
            e[0] = np.log(e[0] + 1)
        else:
            e[0] = np.log(1.0 / (-e[0] + 1))
        K = np.copy(W.reshape(2, 2))
        # if abs(e[0]) > 2 * self.target[0]:
        #     K[0, 0] = 0
        #     K[1, 0] = 0
        u = np.matmul(K, e).reshape(-1) + self.offset
        u[u > self.max] = self.max
        u[u < self.min] = self.min
        return u

    def sampleMat(self, W, X):
        '''
        Inputs:
            W   policy weights                  (4 x N)
            X   vector of states                (N x 2)

        Outputs (N x 2)
        '''
        assert W.shape[0] == 4, 'Wrong policy dimensions'

        e = self.target - X # (10, 2)

        oz = e[:, 0] >= 0
        e[oz, 0] = np.log(e[oz, 0] + 1)
        oz = np.invert(oz)
        e[oz, 0] = np.log(-1.0 / (e[oz, 0] - 1))

        e = e.T.reshape(2, 1, -1)

        K = np.copy(W.reshape(2, 2, -1))

        # K (2 x 2 x N)
        # e (2 x 1 x N)
        u = np.einsum('ijn,jkn->ikn', K, e)[:, 0, :].T + self.offset # (N x 2)
        u[u > self.max] = self.max
        u[u < self.min] = self.min
        return u

class PropDerv:
    def __init__(self, min, max, target, offset):
        self.min = min
        self.max = max
        self.target = target # (2, )
        self.offset = offset
        self.init = False

    def reset(self):
        self.init = False

    def sample(self, W, X):
        '''
        Inputs:
            W   policy weights                  (4 x 1)
            X   vector of states                (2, )
        '''
        e = (self.target - X).reshape(-1, 1) # error
        if(e[0] >= 0):
            e[0] = np.log(e[0] + 1)
        else:
            e[0] = np.log(1.0 / (-e[0] + 1))

        if not self.init:
            self.init = True
            self.prev_e = e

        de = e - prev_e

        emat = np.concatenate(e, de)

        K = np.copy(W.reshape(2, 4))

        u = np.matmul(K, emat).reshape(-1) + self.offset
        u[u > self.max] = self.max
        u[u < self.min] = self.min
        return u

    def sampleMat(self, W, X):
        '''
        Inputs:
            W   policy weights                  (4 x N)
            X   vector of states                (N x 2)

        Outputs (N x 2)
        '''
        assert W.shape[0] == 4, 'Wrong policy dimensions'

        e = self.target - X # (10, 2)

        oz = e[:, 0] >= 0
        e[oz, 0] = np.log(e[oz, 0] + 1)
        oz = np.invert(oz)
        e[oz, 0] = np.log(-1.0 / (e[oz, 0] - 1))

        e = e.T.reshape(2, 1, -1)

        if not self.init:
            self.init = True
            self.prev_e = e

        de = e - self.prev_e
        emat = np.concatenate(e, de)

        K = np.copy(W.reshape(2, 4, -1))

        # K (2 x 2 x N)
        # e (2 x 1 x N)
        u = np.einsum('ijn,jkn->ikn', K, emat)[:, 0, :].T + self.offset # (N x 2)
        u[u > self.max] = self.max
        u[u < self.min] = self.min
        return u

class PropInt:
    def __init__(self, min, max, target, offset, maxI = 0, minI = 0, dt = 1):
        self.min = min
        self.max = max
        self.target = target # (2, )
        self.offset = offset
        self.init = False
        self.maxI = maxI
        self.minI = minI
        self.dt = dt

    def reset(self):
        self.init = False

    def sample(self, W, X):
        '''
        Inputs:
            W   policy weights                  (8 x 1)
            X   vector of states                (2, )
        '''
        # If initialization is needed
        if not self.init:
            self.init = True
            self.I = np.zeros(2)

        # Weights
        Kp = np.copy(W[0:4].reshape(2, 2))
        Ki = np.copy(W[4: ].reshape(2, 2))
        
        # Error
        e = (self.target - X).reshape(-1, 1)
        if(e[0] >= 0):
            e[0] = np.log(e[0] + 1)
        else:
            e[0] = np.log(1.0 / (-e[0] + 1))

        # Integral component
        self.I += np.matmul(Ki, e * self.dt).reshape(-1)
        self.I[self.I > self.maxI] = self.maxI
        self.I[self.I < self.minI] = self.minI

        # Compute and saturate output
        u = np.matmul(Kp, e).reshape(-1) + self.offset + self.I
        u[u > self.max] = self.max
        u[u < self.min] = self.min

        return u

    def sampleMat(self, W, X):
        '''
        Inputs:
            W   policy weights                  (8 x N)
            X   vector of states                (N x 2)

        Outputs (N x 2)
        '''
        # Prevent errors later on when reshaping weights
        assert W.shape[0] == 8, 'Wrong policy dimensions'

        # If initialization is needed
        if not self.init:
            self.init = True
            self.I = np.zeros((X.shape[0], 2)) # (N, 2)

        # Weights
        Kp = np.copy(W[0:4, :].reshape(2, 2, -1))
        Ki = np.copy(W[4: , :].reshape(2, 2, -1))

        # Error
        e = self.target - X
        oz = e[:, 0] >= 0
        e[oz, 0] = np.log(e[oz, 0] + 1)
        oz = np.invert(oz)
        e[oz, 0] = np.log(-1.0 / (e[oz, 0] - 1))
        e = e.T.reshape(2, 1, -1)

        # Integral component
        self.I += np.einsum('ijn,jkn->ikn', Ki, e * self.dt)[:, 0, :].T
        self.I[self.I > self.maxI] = self.maxI
        self.I[self.I < self.minI] = self.minI

        # Compute output
        u = np.einsum('ijn,jkn->ikn', Kp, e)[:, 0, :].T + self.offset + self.I
        u[u > self.max] = self.max
        u[u < self.min] = self.min

        return u

if __name__ == "__main__":
#     pol = TilePol(5, 1, -2, 10, -2)
#     ws = np.array([[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5]])
#     print pol.sample(ws , np.array([-3, -1, 2]).reshape(-1,1))
    tgt = np.array([10, 0]).reshape(-1)
    off = np.array([150, 150]).reshape(-1)
    # pol = Proportional(-326, 326, tgt, off)
    # w0 = np.array([10, 0, 10, 0]).reshape(-1)
    # print pol.sample(w0, np.array([0, 0]))
