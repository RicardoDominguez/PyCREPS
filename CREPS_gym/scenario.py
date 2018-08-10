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

    def reset(self):
        '''
        Call this at the start of an episode.
        '''
        self.init = False

    def sample(self, W, X):
        '''
        Inputs:
            W   policy weights                  (4 x 1)
            X   vector of states                (1 x 4)

        Outputs:
            u   discrete control action         (1 x 1)
        '''
        if X.ndim == 1: X = X.reshape(1, -1)
        return self.discretize(X.dot(W))

    def discretize(self, u):
        '''
        Discretize control output for compatibility with gym CartPole environment
        '''
        if u < 0:
            return 0
        else:
            return 1


def predictReward(env, M, hipol, pol):
    '''
    Perform exploration before policy update.

    Inputs:
        env     gym environment
        hipol   upper-level policy
        pol     lower-level policy

    Outputs:
        R       reward for each episode                         (M,  )
        W       lower-level weights used for each episode       (M x nW)
        F       context of each episode                         (M x nS)
    '''
    R = np.zeros(M)
    W = np.empty((M, 4))
    F = np.empty((M, 4))

    for rollout in xrange(M):
        s = env.reset()                    # Sample context
        w = hipol.sample(s.reshape(1, -1)) # Sample lower-policy weights

        W[rollout, :] = w
        F[rollout, :] = s

        done = False
        x = s
        while not done:
            u = pol.sample(w.T, x)
            x, r, done, info = env.step(u)
            R[rollout] += r

    return R, W, F
