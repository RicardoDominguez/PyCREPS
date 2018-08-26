""" Implementation of functions classes relevant to a specific learning scenario.
    This file implements those relevant to the wall following robot scenario.
"""

import numpy as np

class LowerPolicy:
    '''
    PD controller with discrete output for gym CartPole environment
    '''
    def __init__(self, target):
        '''
        Inputs:
            target      (nO, )
        '''
        self.target = target # To compute error

    def sample(self, W, X):
        '''
        Inputs:
            W   policy weights                  (N x 4)
            X   vector of states                (N x 4)

        Outputs (N x 1)
        '''
        if X.ndim == 1: X = X.reshape(1, -1)
        u = np.einsum('ij,ij->i', W, X).reshape(-1, 1)
        return self.discretize(u)

    def discretize(self, u):
        '''
        Discretize control output for compatibility with gym CartPole environment
        '''
        neg = u < 0
        u[neg]            = 0 # Negative output to 0
        u[np.invert(neg)] = 1 # Positive output to 1
        if u.shape[0] == 1: return int(u[0, 0])
        return u

def sampleContext(N):
    '''
    Samples N random contexts.
    '''
    return np.random.uniform(low=-0.05, high=0.05, size=(N, 4))

def systemRollout(env, hipol, pol):
    latent = np.empty((201, 5))

    x = env.reset()                    # Sample context
    w = hipol.mean(x.reshape(1, -1)).T # Sample lower-policy weights

    h = 0
    done = False
    while not done:
        u = pol.sample(w, x)
        latent[h, :] = np.append(x, u)
        x, r, done, info = env.step(u)
        h += 1
    latent[h, :] = np.append(x, u)

    X = latent[0:h,   :]
    Y = latent[1:h+1, :]
    return X, Y

def predictReward(M, hipol, pol, gp):
    '''
    Perform exploration before policy update.

    Inputs:
        M       number of rollouts
        hipol   upper-level policy
        pol     lower-level policy
        gp      GP model

    Outputs:
        R       reward for each episode                         (M,  )
        W       lower-level weights used for each episode       (M x nW)
        F       context of each episode                         (M x nS)
    '''
    F = sampleContext(M) # Draw context
    W = hipol.sample(F)  # Draw lower-policy parameters
    R = np.zeros((M,1))

    H = 0
    not_done = np.ones(M, dtype = 'bool')
    x = F
    while not_done.any() and H < 200:
        H += 1

        u = pol.sample(W[not_done, :], x[not_done, :])
        xu = np.concatenate((x[not_done, :], u), 1)
        x[not_done, :] = gp.predict(xu)

        R[not_done] += 1

        not_done[not_done] = np.logical_and(abs(x[not_done, 0]) <= 2.4, abs(x[not_done, 2] <= np.pi / 15))

    return R, W, F
