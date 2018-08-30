""" Main scrip illustrating the use of Contextual REPS to learn an upper-level
    policy for gym CartPole environment.
"""

# Allow import of CREPS.py module in upper directory
import sys
import os.path
sys.path.append( os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

# Imports needed
import gym
import time
import torch
torch_type = torch.double
from CREPS_torch    import computeSampleWeighting, UpperPolicy
from scenario       import LowerPolicy, predictReward # Scenario specific
from benchmarks     import bench

# ------------------------------------------------------------------------------
# Contextual REPS algorithm parameters
# ------------------------------------------------------------------------------
eps = 1            # Relative entropy bound (lower -> more exploration)
M = 100            # Number of rollouts per policy iteration

# -----------------------------------------------------------------------------
# Scenario parameters
# -----------------------------------------------------------------------------
upper_a = torch.tensor([[1., 0., 1., 0.]], dtype = torch_type) # Initial upper-policy parameters
upper_A = torch.zeros((4, 4), dtype = torch_type)
upper_sigma = torch.eye(4, dtype = torch_type) * torch.tensor([.1, .1, .1, .1], dtype = torch_type)

# ------------------------------------------------------------------------------
# Initialization of necesary classes
# ------------------------------------------------------------------------------
pol = LowerPolicy()
hpol = UpperPolicy(4, torchOut = True)
hpol.set_parameters(upper_a, upper_A, upper_sigma)
env = gym.make('CartPole-v0')

# ------------------------------------------------------------------------------
# Allow consistent results
# ------------------------------------------------------------------------------
env.seed(10)
torch.manual_seed(2)

# Benchmark of initial policy
print('--------------------------------------')
print('Initial policy...')
muR, solved = bench(env, hpol, pol, True)

# ------------------------------------------------------------------------------
# Policy iteration
# ------------------------------------------------------------------------------
k = 0
total_time = 0
while not solved:
    print('--------------------------------------')
    print('Run', k+1)
    k += 1

    R, W, F = predictReward(env, M, hpol, pol)

    s = time.time()

    p = computeSampleWeighting(R, F, eps)
    hpol.update(W, F, p)

    t = time.time() - s
    print("Update time", t)
    total_time += t

    muR, solved = bench(env, hpol, pol, True)

print("Average update time", total_time/k)
