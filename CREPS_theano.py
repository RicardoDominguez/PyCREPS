"""Numpy implementation of:
        -The CREPS optimizer
        -A linear-Gaussian upper-level policy
"""

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from numpy.random import multivariate_normal as mvnrnd
import theano
import theano.tensor as T


def compileDualFunction():
    x = T.dvector('x')
    R = T.dmatrix('R')
    F = T.dmatrix('F')
    eps = T.dscalar('eps')

    eta = x[0]
    theta = x[1:].reshape((-1, 1))

    F_mean = F.mean(0).reshape((1, -1))
    R_over_eta = (R - F.dot(theta)) / eta
    R_over_eta_max = R_over_eta.max()
    Z = T.exp(R_over_eta - R_over_eta_max).T
    Z_sum = Z.sum()
    log_sum_exp = R_over_eta_max + T.log(Z_sum / F.shape[0])

    f = eta * (eps + log_sum_exp) + F_mean.dot(theta)
    d_eta = eps + log_sum_exp - (Z.dot(R_over_eta) / Z_sum)
    d_theta = F_mean - (Z.dot(F) / Z_sum)

    return theano.function([x, R, F, eps], [f, d_eta, d_theta])

def compileSampleWeights():
    x = T.dvector('x')
    R = T.dmatrix('R')
    F = T.dmatrix('F')

    eta = x[0]
    theta = x[1:].reshape((-1, 1))

    R_baseline_eta = (R - F.dot(theta)) / eta
    p = T.exp(R_baseline_eta - R_baseline_eta.max())
    p_s = p / p.sum()

    return theano.function([x, R, F], p_s)

def compileLinGaussMean():
    a = T.dmatrix('a')
    A = T.dmatrix('A')
    S = T.dmatrix('S')

    mu = a + S.dot(A)

    return theano.function([a, A, S], mu)

def compileMLUpdate():
    S = T.dmatrix('S')
    W = T.dmatrix('W')
    p = T.dvector('p')

    P = T.basic.diag(p)

    # Compute new mean
    bigA = T.nlinalg.pinv(S.T.dot(P).dot(S)).dot(S.T).dot(P).dot(W)
    a = bigA[0, :]
    A = bigA[1:, :]

    # Compute new covariance matrix
    wd = W - a
    sigma = (p * wd.T).dot(wd)

    return theano.function([S, W, p], [a, A, sigma])

t_dual_fnc = compileDualFunction()
t_samp_weights = compileSampleWeights()
t_lin_gauss_mean = compileLinGaussMean()
t_ML_update = compileMLUpdate()

def computeSampleWeighting(R, F, eps):
    """Compute sample weights for the upper-level policy update.

    Computes the sample weights used to update the upper-level policy, according
    to the set of features and rewards found by interacting with the environment.

    Parameters
    ----------

    R: numpy.ndarray, shape (n_samples, 1)
        Rewards

    F: numpy.ndarray, shape (n_samples, n_context_features)
        Context features

    eps: float
        Epsilon

    Returns
    -------

    p: numpy.ndarray, shape (n_samples,)
        Weights for policy update
    """
    assert(R.shape[1] == 1 and
           R.shape[0] == F.shape[0]
           ), "Incorrect parameter size"
    # ----------------------------------------------------------------------
    # Minimize dual function using L-BFGS-B
    # ----------------------------------------------------------------------
    def dual_fnc(x): # Dual function with analyitical gradients
        f, deta, dtheta = t_dual_fnc(x, R, F, eps)
        return f, np.append(deta, dtheta)

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
    return t_samp_weights(x, R, F).reshape(-1,)


class UpperPolicy:
    """Upper-level policy.

    Upper-level policy \pi(w | s) implemented as a linear-Gaussian model
    parametrized by {a, A, sigma}:
            \pi(w | s) = N(w | a + As, sigma)

    Parameters
    ----------

    n_context: int
        Number of context features

    verbose: bool, optional (default: False)
        If True prints the policy parameters after a policy update
    """
    def __init__(self, n_context, verbose = False):
        self.n_context = n_context
        self.verbose = verbose

    def set_parameters(self, a, A, sigma):
        """Set the paramaters of the upper-level policy.

        Parameters
        ----------

        a: numpy.ndarray, shape (1, n_lower_policy_weights)
            Parameter 'a'

        A: numpy.ndarray, shape (n_context_features, n_lower_policy_weights)
            Parameter 'A'

        sigma: numpy.ndarray, shape (n_lower_policy_weights,
                                    n_lower_policy_weights)
            Covariance matrix
        """
        n_lower_policy_weights = a.shape[1]
        assert(a.shape[0] == 1 and
               A.shape[1] == n_lower_policy_weights and
               A.shape[0] == self.n_context and
               sigma.shape[0] == n_lower_policy_weights and
               sigma.shape[1] == n_lower_policy_weights
               ), "Incorrect parameter sizes"
        self.a = a
        self.sigma = sigma
        self.A = A

    def sample(self, S):
        """Sample the upper-level policy given the context features.

        Sample distribution \pi(w | s) = N(w | a + As, sigma)

        Parameters
        ----------

        S: numpy.ndarray, shape (n_samples, n_context_features)
            Context features

        Returns
        -------

        W: numpy.ndarray, shape (n_samples, n_lower_policy_weights)
           Sampled lower-policy parameters.
        """
        W = np.zeros((S.shape[0], self.a.shape[1]))
        mus = self.mean(S)
        for sample in range(S.shape[0]):
            W[sample, :] = mvnrnd(mus[sample, :], self.sigma)
        return W

    def mean(self, S):
        """Return the upper-level policy mean given the context features.

        The mean of the distribution is N(w | a + As, sigma)

        Parameters
        ----------

        S: numpy.ndarray, shape (n_samples, n_context_features)
            Context features

        Returns
        -------

        W: numpy.ndarray, shape (n_samples, n_lower_policy_weights)
           Distribution mean for contexts
        """
        return t_lin_gauss_mean(self.a, self.A, S)

    def update(self, w, F, p):
        """Update the upper-level policy parametersself.

        Update is done using weighted maximum likelihood.

        Parameters
        ----------

        w: numpy.ndarray, shape (n_samples, n_lower_policy_weights)
            Lower-level policy weights

        F: numpy.ndarray, shape (n_samples, n_context_features)
            Context features

        p: numpy.ndarray, shape (n_samples,)
            Sample weights
        """
        n_samples = w.shape[0]
        n_lower_policy_weights = self.a.shape[1]
        assert(w.shape[1] == n_lower_policy_weights and
               F.shape[0] == n_samples and
               F.shape[1] == self.n_context and
               p.shape[0] == n_samples and
               p.ndim == 1
               ), "Incorrect parameter size"

        S = np.concatenate((np.ones((p.size, 1)), F), axis = 1)
        a, A, sigma = t_ML_update(S, w, p)

        # Update policy parameters
        self.set_parameters(a.reshape(1,-1), A, sigma)

        if self.verbose:
            print('Policy update: a, A, mean of sigma')
            print(self.a)
            print(self.A)
            print(self.sigma.mean())
