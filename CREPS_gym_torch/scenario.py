""" Implementation of functions classes relevant to a specific learning scenario.
    This file implements those relevant to the wall following robot scenario.
"""

import torch
torch_type = torch.double

class LowerPolicy:
    """
    Linear policy with discrete output for gym CartPole environment
    """
    def sample(self, W, X):
        """Sample lower-level policy.

        Parameters
        ----------

        W: torch.Tensor, shape (4, 1)
            Policy weights

        X: torch.Tensor, shape (1, 4)
            Agent state

        Returns
        -------

        u: float
            Control action
        """
        return self.discretize(float(X.mm(W)))

    def discretize(self, u):
        """Discretize control action (compatible with gym environment)

        Parameters
        ----------

        u: float
            Control action

        Returns
        -------

        0 or 1: int
            Discrete control action
        """
        if u < 0:
            return 0
        else:
            return 1


def predictReward(env, n_episodes, hipol, pol):
    """Interact with environment (before policy update)

    Parameters
    ----------

    env: gym environment

    n_episodes: int
        Number of episodes

    hipol: UpperPolicy

    pol: LowerPolicy

    Returns
    -------

    R: torch.Tensor, shape (n_episodes, 1)
        Reward of each episode

    W: torch.Tensor, shape (n_episodes, n_lower_policy_weights)
        Lower-policy weights for each episode

    F: torch.Tensor, shape (n_episodes, n_context_features)
        Context features for each episode
    """
    R = torch.zeros((n_episodes, 1), dtype = torch_type)
    W = torch.empty((n_episodes, 4), dtype = torch_type)
    F = torch.empty((n_episodes, 4), dtype = torch_type)

    for rollout in range(n_episodes):
        x = torch.from_numpy(env.reset()).view(1, -1)
        w = hipol.sample(x) # Sample lower-policy weights

        W[rollout, :] = w
        F[rollout, :] = x

        done = False
        while not done:
            u = pol.sample(w.t(), x)
            x, r, done, info = env.step(u)
            x = torch.from_numpy(x).view(1,-1)
            R[rollout] += r

    return R, W, F
