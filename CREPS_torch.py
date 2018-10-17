"""PyTorch implementation of the CREPS optimizer and upper-level policy.

The numpy implementation will generally be faster for relatively small problems.
Check that your application is sufficiently involved to benefit computationally.
"""

import numpy as np
import torch
from scipy.optimize import fmin_l_bfgs_b
from torch.distributions.multivariate_normal import MultivariateNormal
torch_type = torch.double


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

    p: torch.Tensor, shape (n_samples,)
        Weights for policy update
    """
    assert(R.shape[1] == 1 and
           R.shape[0] == F.shape[0]
           ), "Incorrect parameter size"

    if type(R).__module__ == np.__name__:
        R = torch.from_numpy(R).view(-1,1)
    if type(F).__module__ == np.__name__:
        F = torch.from_numpy(F)
    # ----------------------------------------------------------------------
    # Minimize dual function using L-BFGS-B
    # ----------------------------------------------------------------------
    def dual_fnc(x_): # Dual function with analyitical gradients
        x = torch.as_tensor(x_)

        eta = x[0]
        theta = x[1:].view(-1,1)

        F_mean = F.mean(0).view(1,-1)
        R_over_eta = (R - F.mm(theta)) / eta
        R_over_eta_max = R_over_eta.max()
        Z = torch.exp(R_over_eta - R_over_eta_max)
        Z_sum = Z.sum()
        log_sum_exp = R_over_eta_max + torch.log(Z_sum / F.shape[0])

        f = eta * (eps + log_sum_exp) + F_mean.mm(theta)

        d_eta = eps + log_sum_exp - Z.t().mm(R_over_eta)/Z_sum
        d_theta = F_mean - (Z.t().mm(F) / Z_sum)
        return f.numpy(), np.append(d_eta.numpy(), d_theta.numpy())

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

    theta = torch.from_numpy(theta).view(-1,1)
    R_baseline_eta = (R - F.mm(theta)) / eta
    p = torch.exp(R_baseline_eta - R_baseline_eta.max()).view(-1)
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

    torchOut: bool, optional (default: True)
        If True the policy returns torch tensors, otherwise numpy arrays

    verbose: bool, optional (default: False)
        If True prints the policy parameters after a policy update
    """
    def __init__(self, n_context, torchOut = True, verbose = False):
        self.n_context = n_context
        self.torchOut = torchOut
        self.verbose = verbose

    def set_parameters(self, a, A, sigma):
        """Set the paramaters of the upper-level policy.

        Parameters
        ----------

        a: numpy.ndarray or torch.Tensor, shape (1, n_lower_policy_weights)
            Parameter 'a'

        A: numpy.ndarray or torch.Tensor, shape (n_context_features,
                                                n_lower_policy_weights)
            Parameter 'A'

        sigma: numpy.ndarray or torch.Tensor, shape (n_lower_policy_weights,
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

        if type(a).__module__ == np.__name__: #Assume all same type
            self.a = torch.from_numpy(a)
            self.A = torch.from_numpy(A)
            self.sigma = torch.from_numpy(sigma)
        else:
            self.a = a
            self.A = A
            self.sigma = sigma
        self.mvnrnd = MultivariateNormal(self.a.view(-1), self.sigma)

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
        if type(S).__module__ == np.__name__:
            S = torch.from_numpy(S)

        W = torch.zeros(S.shape[0], self.a.shape[1], dtype = torch_type)
        mus = self.mean(S)

        if not self.torchOut:
              mus = torch.from_numpy(mus)

        for sample in range(S.shape[0]):
            self.mvnrnd.loc = mus[sample, :]
            W[sample, :] = self.mvnrnd.sample()

        if self.torchOut:
            return W
        else:
            return W.numpy()

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
        if type(S).__module__ == np.__name__:
            S = torch.from_numpy(S)

        mu = self.a + S.mm(self.A)

        if self.torchOut:
            return mu
        else:
            return mu.numpy()

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

        p: torch.Tensor, shape (n_samples,)
            Sample weights
        """
        n_samples = w.shape[0]
        n_lower_policy_weights = self.a.shape[1]
        assert(w.shape[1] == n_lower_policy_weights and
               F.shape[0] == n_samples and
               F.shape[1] == self.n_context and
               p.shape[0] == n_samples
               )

        if type(F).__module__ == np.__name__:
            F = torch.from_numpy(F)
        if type(w).__module__ == np.__name__:
            w = torch.from_numpy(w)

        S = torch.cat((torch.ones(p.shape[0], 1, dtype = torch_type), F), 1)
        P = p.diag()
        bigA = torch.pinverse(S.t().mm(P).mm(S)).mm(S.t()).mm(P).mm(w)
        a = bigA[0, :].view(1, -1)

        wd = w - a
        sigma = (p * wd.t()).mm(wd)

        self.set_parameters(a, bigA[1:, :], sigma)

        if self.verbose:
            print('Policy update: a, A, mean of sigma')
            print(self.a)
            print(self.A)
            print(self.sigma.mean())
