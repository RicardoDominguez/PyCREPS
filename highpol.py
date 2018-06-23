import numpy as np
import numpy.random.multivariate_normal as mvnrnd

# Defines the high level policy as a multivariate normal distribution
class HighPol:
    def __init__(self, mu, sigma):
        '''
        Inputs:
            mu      parameter mean
            sigma   parameter covariance matrix
        '''
        self.mu = mu            # (W x 1)
        self.sigma = sigma      # (W x W)

    def sample(self, N = 1):
        '''
        Sample from a multivariate normal distribution

        Inputs:
            N   number of samples

        Outputs: vector of sampled control actions
        '''
        return mvnrnd(self.mu, self.sigma, (N, 1))

    def update(self, p, w):
        '''
        Update high level policy mean and covariance matrix using weighted
        maximum likelihood

        Inputs:
            p   weights for weighted ML                     (N x 1)
            w   low level policy parameters from samples    (N x W)
        '''
        N = p.size()
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
            nom = ps[i] * mu_diff * mu_diff.T
            sum_sigma += norm

        # Update mean and sigma
        self.mu = mu
        self.sigma = sum_sigma
