""" Implementation of functions classes relevant to a specific learning scenario.
    This file implements those relevant to the wall following robot scenario.
"""

import numpy as np


class LowerPolicy:
    '''
    Linear controller with discrete output for gym Acrobot environment
    '''
    def sample(self, W, X):
        '''
        Inputs:
            W   policy weights                  (6 x 1)
            X   vector of states                (1 x 6)

        Outputs:
            u   discrete control action         (1 x 1)
        '''
        return self.discretize(X.dot(W).flatten())

    def discretize(self, u):
        '''
        Discretize control output for compatibility with gym CartPole environment
        '''
        if u < 1:
            return 0 # -1
        elif u > 1:
            return 2 # 1
        else:
            return 1 # 0


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
    R = np.zeros((M, 1))
    W = np.empty((M, 6))
    F = np.empty((M, 2))

    for rollout in range(M):
        s = env.reset()                    # Sample context

        F[rollout, :] = s[-2:]

        w = hipol.sample(F[rollout, :].reshape(1, -1)) # Sample lower-policy weights

        W[rollout, :] = w

        done = False
        x = s
        while not done:
            u = pol.sample(w.T, x)
            x, r, done, info = env.step(u)
            R[rollout] += r

    R -= R.min() # Make all rewards positive
    return R, W, F
