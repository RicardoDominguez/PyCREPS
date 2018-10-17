"""Theano implementation of the CREPS optimizer and upper-level policy.

The numpy implementation will generally be faster for relatively small problems.
Check that your application is sufficiently involved to benefit computationally.
"""

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from numpy.random import multivariate_normal as mvnrnd
import theano
import theano.tensor as T

class dualFunction:
    """Used to compute dual function and its derivative.

    Using this class allows to set shared theano variables, improving run time
    performance.

    Theano functions:
        self.sample_f: evaluate dual function at x and its gradient
        self.sample_weights: evaluate weights for weighted ML update
        self.updateGauss: update upper level parameters a, A, S
    """
    def __init__(self):
        self.initRFeps = False
        self.initGauss = False

    def updateSharedVarRFeps(self, R, F, eps):
        """Update the shared variables R, F and eps.

        If the variables have not been initialised (self.initRFeps), the shared
        variables R, F and eps and the variable x are also initialized.

        Parameters
        ----------

        R: numpy.ndarray, shape (n_samples, 1)
            Rewards

        F: numpy.ndarray, shape (n_samples, n_context_features)
            Context features

        eps: float
            Epsilon
        """
        if self.initRFeps: # Update variables
            self.R.set_value(R)
            self.F.set_value(F)
            self.eps.set_value(eps)
        else: # Initialise if needed
            self.x = T.dvector('x')
            self.R = theano.shared(R)
            self.F = theano.shared(F)
            self.eps = theano.shared(eps)
            self.setLossGrad()
            self.initRFeps = True

    def updateSharedGauss(self, a, A):
        """Update the shared variables a, A and S.

        If the variables have not been initialised (self.initGauss), the shared
        variables R, F and eps and the variable x are also initialized.

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
        if self.initGauss: # Update variables
            self.a.set_value(a)
            self.A.set_value(A)
        else: # Initialise if needed
            self.a = theano.shared(a)
            self.A = theano.shared(A)
            self.setGauss()
            self.initGauss = True

    def setGauss(self):
        """Set the theano function to compute the mean for Gaussian sampling.

        This function only needs to be called once (not every policy update).

        The theano variables self.a and self.A must be defined previous
        to this function call.
        """
        S = T.dmatrix('S')
        mu = self.a + S.dot(self.A)
        self.gaussMean = theano.function([S], mu)

    def setLossGrad(self):
        """Set the theano function to compute dual functions and Gauss update.

        This function only needs to be called once (not every policy update).

        The theano variables self.x, self.F, self.R, self.eps, self.a and
        self.A must be defined previous to this function call.
        """
        # ---------------------------------------------------------------------
        # Evaluate dual function at x and its gradient.
        # ---------------------------------------------------------------------
        eta = self.x[0]
        theta = self.x[1:].reshape((-1, 1))

        F_mean = self.F.mean(0).reshape((1, -1))
        R_over_eta = (self.R - self.F.dot(theta)) / eta
        R_over_eta_max = R_over_eta.max()
        Z = T.exp(R_over_eta - R_over_eta_max).T
        Z_sum = Z.sum()
        log_sum_exp = R_over_eta_max + T.log(Z_sum / self.F.shape[0])

        # f wrapped in mean to prevent "cost must be a scalar" error
        f = T.mean(eta * (self.eps + log_sum_exp) + F_mean.dot(theta))
        d_x = T.grad(f, self.x)
        self.sample_f = theano.function([self.x], [f, d_x])

        # ---------------------------------------------------------------------
        # Sample weights for weighted ML update.
        # ---------------------------------------------------------------------
        p = Z / Z_sum
        self.sample_weights = theano.function([self.x], p)

        # ---------------------------------------------------------------------
        # Upper level policy update.
        # ---------------------------------------------------------------------
        S = T.dmatrix('S')
        W = T.dmatrix('W')
        p_ = T.dvector('p_')

        # Compute means
        P = T.basic.diag(p_)
        bigA = T.nlinalg.pinv(S.T.dot(P).dot(S)).dot(S.T).dot(P).dot(W)
        a = bigA[0, :]
        A = bigA[1:, :]

        # Compute new covariance matrix
        wd = W - a
        sigma = (p_ * wd.T).dot(wd)
        self.updateGauss = theano.function([S, W, p_], [a, A, sigma])

t_dualFnc = dualFunction()

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

    # Update theano shared variables
    t_dualFnc.updateSharedVarRFeps(R, F, eps)

    # Initial point
    x0 = [1] + [1] * F.shape[1]

    # Bounds
    min_eta = 1e-10
    bds = np.vstack(([[min_eta, None]], np.tile(None, (F.shape[1], 2))))

    # Minimize using L-BFGS-B algorithm
    x = fmin_l_bfgs_b(t_dualFnc.sample_f, x0, bounds=bds)[0]

    # Return weights of individual samples for policy update
    return t_dualFnc.sample_weights(x).reshape(-1,)


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
        self.n_lower_policy_weights = a.shape[1]
        assert(a.shape[0] == 1 and
               A.shape[1] == self.n_lower_policy_weights and
               A.shape[0] == self.n_context and
               sigma.shape[0] == self.n_lower_policy_weights and
               sigma.shape[1] == self.n_lower_policy_weights
               ), "Incorrect parameter sizes"
        t_dualFnc.updateSharedGauss(a, A)
        self.sigma = sigma

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
        W = np.zeros((S.shape[0], self.n_lower_policy_weights))
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
        return t_dualFnc.gaussMean(S)

    def update(self, w, F, p):
        """Update the upper-level policy parameters.

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
        assert(w.shape[1] == self.n_lower_policy_weights and
               F.shape[0] == n_samples and
               F.shape[1] == self.n_context and
               p.shape[0] == n_samples and
               p.ndim == 1
               ), "Incorrect parameter size"

        S = np.concatenate((np.ones((p.size, 1)), F), axis = 1)
        a, A, sigma = t_dualFnc.updateGauss(S, w, p)

        # Update policy parameters
        self.set_parameters(a.reshape(1,-1), A, sigma)

        if self.verbose:
            print('Policy update: a, A, mean of sigma')
            print(self.a)
            print(self.A)
            print(self.sigma.mean())
