""" Implementation of the CREPS optimizer and linear-Gaussian upper-level policy """

import numpy as np
from scipy.misc import logsumexp
from scipy.optimize import fmin_l_bfgs_b
from numpy.random import multivariate_normal as mvnrnd

def computeSampleWeighting(R, F, eps):
    '''
    Computes the sample weights used to update the upper-level policy, according
    to the set of features and rewards found by interacting with the model.

    Inputs:
        R       Return  dataset vector  (N ,  )
        F       Feature dataset matrix  (N x nS)
        eps     Epsilon                 (1 x 1)

    Outputs:
        p       policy update weights   (N x 1)
    '''
    # ----------------------------------------------------------------------
    # Minimize dual function using L-BFGS-B
    # ----------------------------------------------------------------------
    def dual_fnc(x): # Dual function with analyitical gradients
        eta = x[0]
        theta = x[1:]

        F_mean = F.mean(0)
        R_over_eta = (R - theta.dot(F.T)) / eta
        log_sum_exp = logsumexp(R_over_eta, b = 1.0 / F.shape[0])
        Z = np.exp(R_over_eta - R_over_eta.max())
        Z_sum = Z.sum()

        f = eta * (eps + log_sum_exp) + theta.T.dot(F_mean)
        d_eta = eps + log_sum_exp - (Z.dot(R_over_eta) / Z_sum)
        d_theta = F_mean - (Z.dot(F) / Z_sum)
        return f, np.append(d_eta, d_theta)

    # Initial point
    x0 = [1] + [1] * F.shape[1]

    # Bounds
    min_eta = 1e-10
    bds = np.vstack(([[min_eta, None]], np.tile(None, (F.shape[1], 2))))

    # Minimize using L-BFGS-B algorithm
    x = fmin_l_bfgs_b(dual_fnc, x0, bounds=bds)[0]

    # ----------------------------------------------------------------------
    # Determine weights of individual samples for policy update
    # ----------------------------------------------------------------------
    eta = x[0]
    theta = x[1:]

    R_baseline_eta = (R - theta.dot(F.T)) / eta
    p = np.exp(R_baseline_eta - R_baseline_eta.max())
    p /= p.sum()

    return p

class UpperPolicy:
    '''
    Upper-level policy \pi(w | s) implemented as a linear-Gaussian model
    parametrized by {a, A, sigma}:
            \pi(w | s) = N(w | a + As, sigma)
    '''
    def __init__(self, a0, sigma0, A0 = None, nS = 0, verbose = False):
        '''
        Inputs:
            a0       initial parameter 'a'                      (W, )
            sigma0   initial covariance matrix                  (W x W)
            A0       initial parameter 'A'                      (W x nS)
            nS       context size
            verbose  prints additional information
        '''
        self.a = a0
        self.sigma = sigma0
        if A0 is None:
            self.A = np.zeros((a0.shape[0], nS))
        else:
            self.A = A0
        self.verbose = verbose

    def sample(self, S):
        '''
        Sample upper-level policy

        Inputs:
            S   context                         (N x nS)

        Outputs:
            W   lower-level policy weights      (N x  W)
        '''
        W = np.zeros((S.shape[0], self.a.shape[0]))
        for i in xrange(S.shape[0]):
            W[i, :] = mvnrnd(self.a + np.dot(self.A, S[i, :].T), self.sigma)
        return W

    def mean(self, S):
        '''
        Return the linear-Gaussian model mean

        Inputs:
            S   context                         (1 x nS)

        Outputs:
            W   linear-Gaussian model mean      (W x 1)
        '''
        return self.a.reshape(-1,1) + np.dot(self.A, S.T)

    def update(self, w, F, p):
        '''
        Update upper-level policy parameters using weighted maximum likelihood

        Inputs:
            w   lower-level policy weights      (N x W)
            F   features                        (N x nS)
            p   sample Weights                  (N x 1)
        '''
        S = np.asmatrix(np.concatenate((np.ones((p.size, 1)), F), axis = 1))
        B = np.asmatrix(w);
        P = np.asmatrix(np.diag(p.reshape(-1)))

        # Compute new mean
        Amatrix = np.linalg.pinv(S.T * P * S) * S.T * P * B   # (1 x W)
        a = Amatrix[0, :].reshape(-1, 1)

        # Compute new covariance matrix
        w_mu_diff = w.T - a  # (W x N)
        new_sigma = np.zeros((a.size, a.size)) # (W x W)
        for i in xrange(p.size): new_sigma += p[i] * w_mu_diff[:, i] * w_mu_diff[:, i].T

        # Update policy parameters
        self.a = np.squeeze(np.asarray(a))
        self.A = np.squeeze(np.asarray(Amatrix[1:, :].T))
        self.sigma = new_sigma

        if self.verbose:
            print 'a updated: ', self.a
            print 'A updated: ', self.A
            print 'Sigma updated: ', np.mean(self.sigma)
