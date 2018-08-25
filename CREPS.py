""" Implementation of the CREPS optimizer and linear-Gaussian upper-level policy """

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import pdb

use_pytorch = True
if use_pytorch:
    import torch
    torch_type = torch.double
    from torch.distributions.multivariate_normal import MultivariateNormal
else:
    from numpy.random import multivariate_normal as mvnrnd
    from scipy.special import logsumexp


def computeSampleWeighting(R, F, eps):
    """Compute sample weights for the upper-level policy update.

    Computes the sample weights used to update the upper-level policy, according
    to the set of features and rewards found by interacting with the model.

    Parameters
    ----------

    R: numpy.ndarray or torch.Tensor, shape (n_samples, 1)
        Rewards

    F: numpy.ndarray or torch.Tensor, shape (n_samples, n_context_features)
        Context features

    eps: float
        Epsilon

    Returns
    -------

    p: numpy.ndarray or torch.Tensor, shape (n_samples,)
        Weights for policy update
    """
    if use_pytorch:
        if type(R).__module__ == np.__name__:
            R = torch.from_numpy(R).view(-1,1)
        if type(F).__module__ == np.__name__:
            F = torch.from_numpy(F)
    # ----------------------------------------------------------------------
    # Minimize dual function using L-BFGS-B
    # ----------------------------------------------------------------------
    def dual_fnc(x): # Dual function with analyitical gradients
        eta = x[0]
        if use_pytorch:
            theta = torch.from_numpy(x[1:]).view(-1,1)

            F_mean = F.mean(0).view(1,-1)
            R_over_eta = (R - F.mm(theta)) / eta
            R_over_eta_max = R_over_eta.max()
            Z = torch.exp(R_over_eta - R_over_eta_max)
            Z_sum = Z.sum()
            log_sum_exp = R_over_eta_max + torch.log(Z_sum / F.shape[0])

            # pdb.set_trace()
            f = eta * (eps + log_sum_exp) + F_mean.mm(theta)
            d_eta = eps + log_sum_exp - Z.t().mm(R_over_eta)/Z_sum
            d_theta = F_mean - (Z.t().mm(F) / Z_sum)
            return f.numpy(), np.append(d_eta.numpy(), d_theta.numpy())
        else:
            theta = x[1:]

            F_mean = F.mean(0)
            R_over_eta = (R - theta.dot(F.T)) / eta
            R_over_eta_max = R_over_eta.max()
            Z = np.exp(R_over_eta - R_over_eta_max)
            Z_sum = Z.sum()
            log_sum_exp = R_over_eta_max + np.log(Z_sum / F.shape[0])

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

    if use_pytorch:
        theta = torch.from_numpy(theta).view(-1,1)
        # pdb.set_trace()
        R_baseline_eta = (R - F.mm(theta)) / eta
        p = torch.exp(R_baseline_eta - R_baseline_eta.max()).view(-1)
    else:
        R_baseline_eta = (R - theta.dot(F.T)) / eta
        p = np.exp(R_baseline_eta - R_baseline_eta.max())

    p /= p.sum()
    return p

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

        If PyTorch is being used, the parameters are set as PyTorch tensors.
        Otherwise, they are set as numpy arrays.

        Parameters
        ----------

        a: numpy.ndarray or torch.Tensor, shape (n_lower_policy_weights, 1)
            Parameter 'a'

        A: numpy.ndarray or torch.Tensor, shape (n_lower_policy_weights,
                                                n_context_features)
            Parameter 'A'

        sigma: numpy.ndarray or torch.Tensor, shape (n_lower_policy_weights,
                                                    n_lower_policy_weights)
            Covariance matrix
        """
        if use_pytorch:
            if type(a).__module__ == np.__name__:
                self.a = torch.from_numpy(a)
                self.A = torch.from_numpy(A)
                self.sigma = torch.from_numpy(sigma)
            else:
                self.a = a
                self.A = A
                self.sigma = sigma
            # pdb.set_trace()
            self.mvnrnd = MultivariateNormal(self.a.view(-1), self.sigma)
        else:
            self.a = a
            self.sigma = sigma
            self.A = A

    def sample(self, S):
        """Sample the upper-level policy given the context features.

        Sample distribution \pi(w | s) = N(w | a + As, sigma)

        If PyTorch is being used, the input should be a PyTorch tensor and
        torch.distributions.multivariate_normal is be used, returning a
        tensor. Otherwise, the input should be a numpy array and
        numpy.random.multivariate_normal is used, returning a numpy array.

        Parameters
        ----------

        S: numpy.ndarray or torch.Tensor, shape (n_samples, n_context_features)
            Context features

        Returns
        -------

        W: numpy.ndarray or torch.Tensor, shape (n_samples,
                                                n_lower_policy_weights)
           Sampled lower-policy parameters.
        """
        if use_pytorch:
            if type(S).__module__ == np.__name__:
                S = torch.from_numpy(S)
            W = torch.zeros(S.shape[0], self.a.shape[0])
            for sample in range(S.shape[0]):
                # pdb.set_trace()
                self.mvnrnd.loc = (self.a + self.A.mm(S[sample, :].view(-1,1))).view(-1)
                W[sample, :] = self.mvnrnd.sample()
            return W.numpy()
        else:
            W = np.zeros((S.shape[0], self.a.shape[0]))
            for sample in range(S.shape[0]):
                W[sample, :] = mvnrnd(self.a.flatten() +
                                      self.A.dot(S[sample, :]),
                                      self.sigma)
            return W

    def mean(self, S):
        """Return the upper-level policy mean given the context features.

        The mean of the distribution is N(w | a + As, sigma)

        Parameters
        ----------

        S: numpy.ndarray or torch.Tensor, shape (n_samples, n_context_features)
            Context features

        Returns
        -------

        W: numpy.ndarray or torch.Tensor, shape (n_samples,
                                                n_lower_policy_weights)
           Distribution mean for contexts
        """
        if use_pytorch:
            if type(S).__module__ == np.__name__:
                S = torch.from_numpy(S)

            return (self.a + self.A.mm(S.t())).numpy()
        else:
            return self.a + self.A.dot(S.T)

    def update(self, w, F, p):
        """Update the upper-level policy parametersself.

        Update is done using weighted maximum likelihood.

        Parameters
        ----------

        w: numpy.ndarray or torch.Tensor, shape (n_samples,
                                                n_lower_policy_weights)
            Lower-level policy weights

        F: numpy.ndarray or torch.Tensor, shape (n_samples, n_context_features)
            Context features

        p: numpy.ndarray or torch.Tensor, shape (n_samples,)
            Sample weights
        """
        if use_pytorch:
            if type(F).__module__ == np.__name__:
                F = torch.from_numpy(F)
            if type(w).__module__ == np.__name__:
                w = torch.from_numpy(w)

        if use_pytorch:
            S = torch.cat((torch.ones(p.shape[0], 1, dtype = torch_type), F), 1)
            P = p.diag()
            bigA = torch.pinverse(S.t().mm(P).mm(S)).mm(S.t()).mm(P).mm(w)

            w_mu_diff = w.t() - bigA[0, :].view(-1,1)
            sigma = torch.zeros(w.shape[1], w.shape[1], dtype = torch_type)
            for i in range(p.shape[0]):
                w_diff = w_mu_diff[:,i].view(-1,1)
                sigma.add_(p[i] * w_diff.mm(w_diff.t()))

            self.set_parameters(bigA[0, :].view(-1,1), bigA[1:, :], sigma)
        else:
            S = np.asmatrix(np.concatenate((np.ones((p.size, 1)), F), axis = 1))
            B = np.asmatrix(w);
            P = np.asmatrix(np.diag(p.reshape(-1)))

            # Compute new mean
            bigA = np.linalg.pinv(S.T * P * S) * S.T * P * B
            a = bigA[0, :].reshape(-1, 1)

            # Compute new covariance matrix
            w_mu_diff = w.T - a  # (W x N)
            sigma = np.zeros((a.size, a.size)) # (W x W)
            for i in range(p.size):
                sigma += p[i] * w_mu_diff[:, i] * w_mu_diff[:, i].T

            # Update policy parameters
            self.set_parameters(np.asarray(a).reshape(-1, 1),
                                np.asarray(bigA[1:, :].T), sigma)

        if self.verbose:
            print('Policy update: a, A, mean of sigma')
            print(self.a)
            print(self.A)
            print(self.sigma.mean())
