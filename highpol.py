import numpy as np
from numpy.random import multivariate_normal as mvnrnd
import pdb

# Defines the high level policy as a multivariate normal distribution
class HighPol:
    def __init__(self, mu, sigma, ns = 0, verbose = False):
        '''
        Inputs:
            mu      parameter mean                      (W, )
            sigma   parameter covariance matrix         (W x w)
        '''
        self.mu = mu
        self.sigma = sigma
        self.A = np.zeros((mu.shape[0], ns))
        self.verbose = verbose

    def contextSample(self, S = None, N = 1):
        '''
        Inputs:
            S   context     (N x S)
        '''
        if S is None:
            S = np.zeros((N, A.shape[1]))

        W = np.zeros((self.mu.shape[0], N))
        for i in xrange(N): W[:, i] = mvnrnd(self.mu + np.dot(self.A, S[i, :].T), self.sigma).T
        return W

    def contextMean(self, S):
        '''
        Inputs:
            S   context     (1 x S)

        Output: (W x 1)
        '''
        return self.mu.reshape(-1,1) + np.dot(self.A, S.T)

    def update(self, w, F, p):
        '''
        Update high level policy mean and covariance matrix using weighted
        maximum likelihood

        Inputs:
            w       Weight  dataset matrix  (N x W)
            R       Return  dataset vector  (N x 1)
            F       Feature dataset matrix  (N x S)
        '''
        S = np.asmatrix(np.concatenate((np.ones((p.size, 1)), F), axis = 1)) # Context matrix (N x S + 1)
        B = np.asmatrix(w);  # Parameter matrix (N x W)
        P = np.asmatrix(np.diag(p.reshape(-1)))  # Diagonal weighted matrix  (N x N)

        # Compute mean
        Amatrix = np.linalg.pinv(S.T * P * S) * S.T * P * B   # (1 x W)
        mu = Amatrix[0, :].reshape(-1, 1)

        # Compute sigma
        new_sigma = np.zeros((mu.size, mu.size)) # (W x W)
        w_mu_diff = w.T - mu  # (W x N)
        for i in xrange(p.size):
            mu_diff = w_mu_diff[:, i].reshape(-1, 1)
            nom = p[i] * mu_diff * mu_diff.T
            new_sigma += nom

        # Update mean and sigma
        self.mu = np.squeeze(np.asarray(mu))
        self.A = np.squeeze(np.asarray(Amatrix[1:, :].T))
        self.sigma = new_sigma

        if self.verbose:
            print 'Mean updated', self.mu
            print 'Sigma updated', np.mean(self.sigma)
            print 'A updated', self.A
