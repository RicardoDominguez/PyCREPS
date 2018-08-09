import numpy as np
import pdb

class PID:
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
            W   policy weights                  (12 x 1)
            X   vector of states                (2, )
        '''
        # Weights
        Kp = np.copy(W[0:4].reshape(2, 2))
        Ki = np.copy(W[4:8].reshape(2, 2))
        Kd = np.copy(W[8: ].reshape(2, 2))
        K_p_d = np.concatenate((Kp, Kd), axis = 1)

        # Error
        e = (self.target - X).reshape(-1, 1)
        if(e[0] >= 0):
            e[0] = np.log(e[0] + 1)
        else:
            e[0] = np.log(1.0 / (-e[0] + 1))

        # If initialization is needed
        if not self.init:
            self.init = True
            self.I = np.zeros(2)
            self.prev_e = e

        # Derivative error
        de = (e - self.prev_e) / self.dt
        e_de = np.concatenate((e, de))
        self.prev_e = e

        # Integral component
        self.I += np.matmul(Ki, e * self.dt).reshape(-1)
        self.I[self.I > self.maxI] = self.maxI
        self.I[self.I < self.minI] = self.minI

        # Compute and saturate output
        u = np.matmul(K_p_d, e_de).reshape(-1) + self.offset + self.I
        u[u > self.max] = self.max
        u[u < self.min] = self.min

        return u

    def sampleMat(self, W, X):
        '''
        Inputs:
            W   policy weights                  (12 x N)
            X   vector of states                (N x 2)

        Outputs (N x 2)
        '''
        # Prevent errors later on when reshaping weights
        assert W.shape[0] == 12, 'Wrong policy dimensions'

        # Weights
        Kp = np.copy(W[0:4, :].reshape(2, 2, -1))
        Ki = np.copy(W[4:8, :].reshape(2, 2, -1))
        Kd = np.copy(W[8: , :].reshape(2, 2, -1))
        K_p_d = np.concatenate((Kp, Kd), axis = 1)

        # Error
        e = self.target - X
        oz = e[:, 0] >= 0
        e[oz, 0] = np.log(e[oz, 0] + 1)
        oz = np.invert(oz)
        e[oz, 0] = np.log(-1.0 / (e[oz, 0] - 1))
        e = e.T.reshape(2, 1, -1)

        # If initialization is needed
        if not self.init:
            self.init = True
            self.I = np.zeros((X.shape[0], 2)) # (N, 2)
            self.prev_e = e

        # Derivative error
        de = (e - self.prev_e) / self.dt
        e_de = np.concatenate((e, de))
        self.prev_e = e

        # Integral component
        self.I += np.einsum('ijn,jkn->ikn', Ki, e * self.dt)[:, 0, :].T
        self.I[self.I > self.maxI] = self.maxI
        self.I[self.I < self.minI] = self.minI

        # Compute output
        u = np.einsum('ijn,jkn->ikn', K_p_d, e_de)[:, 0, :].T + self.offset + self.I
        u[u > self.max] = self.max
        u[u < self.min] = self.min

        return u
