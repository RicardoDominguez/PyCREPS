import numpy as np
from numpy.random import multivariate_normal as mvnrnd
from dual_fnc import DualFnc

# Defines the high level policy as a multivariate normal distribution
class HighPol:
    def __init__(self, mu, sigma):
        '''
        Inputs:
            mu      parameter mean                      (W, )
            sigma   parameter covariance matrix         (W x w)
        '''
        self.mu = mu
        self.sigma = sigma
        self.dualFnc = DualFnc()

    def sample(self, N = 1):
        '''
        Sample from a multivariate normal distribution

        Inputs:
            N   number of samples

        Outputs: vector of sampled control actions      (W x N)
        '''
        return (mvnrnd(self.mu, self.sigma, N).T)

    def update(self, w, R, F, eps):
        '''
        Update high level policy mean and covariance matrix using weighted
        maximum likelihood

        Inputs:
            w       Weight  dataset matrix  (N x W)
            R       Return  dataset vector  (N x 1)
            F       Feature dataset matrix  (N x S)
            eps     Epsilon                 (1 x 1)
        '''
        p = self.dualFnc.computeSampleWeighting(w, R, F, eps) # (N x 1)

        N = p.size
        S = np.asmatrix(np.ones((N, 1)))         # Context matrix            (N x 1)
        B = np.asmatrix(w);                      # Parameter matrix          (N x W)
        P = np.asmatrix(np.diag(p.reshape(-1)))  # Diagonal weighted matrix  (N x N)

        # Compute mean
        Amatrix = np.linalg.pinv(S.T * P * S) * S.T * P * B   # (1 x W)
        mu = Amatrix[0, :].reshape(-1, 1)

        # Compute sigma
        W = mu.size
        sum_sigma = np.zeros((W, W))                          # (W x W)
        w_mu_diff = w.T - mu                                  # (W x N)
        ps = p / sum(p)
        for i in xrange(0, N):
            mu_diff = w_mu_diff[:, i].reshape(-1, 1)
            nom = float(ps[i]) * mu_diff * mu_diff.T
            sum_sigma += nom

        # Update mean and sigma
        self.mu = np.squeeze(np.asarray(mu))
        self.sigma = sum_sigma
        print 'Mean updated', self.mu
        print 'Sigma updated', np.mean(self.sigma)
