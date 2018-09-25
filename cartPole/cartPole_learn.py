""" Main scrip illustrating the use of Contextual REPS to learn an upper-level
    policy for gym CartPole environment.
"""

use_torch = False
use_theano = False

# Allow import of CREPS.py module in upper directory
import sys
import os.path
sys.path.append( os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

# Imports needed
import numpy as np
import gym
import time
if use_torch:
    import torch
    from CREPS_torch import computeSampleWeighting, UpperPolicy
    torch.manual_seed(2)
elif use_theano:
    from CREPS_theano import computeSampleWeighting, UpperPolicy
else:
    from CREPS import computeSampleWeighting, UpperPolicy
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
upper_a = np.array([[1., 0., 1., 0.]]) # Initial upper-policy parameters
upper_A = np.zeros((4, 4))
upper_sigma = np.eye(4) * [.1, .1, .1, .1]

# ------------------------------------------------------------------------------
# Initialization of necesary classes
# ------------------------------------------------------------------------------
pol = LowerPolicy()
if use_torch:
    hpol = UpperPolicy(4, torchOut = False)
else:
    hpol = UpperPolicy(4)
hpol.set_parameters(upper_a, upper_A, upper_sigma)
env = gym.make('CartPole-v0')

# ------------------------------------------------------------------------------
# Allow consistent results
# ------------------------------------------------------------------------------
env.seed(10)
np.random.seed(0)

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