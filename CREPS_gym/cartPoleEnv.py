""" Main scrip illustrating the use of Contextual REPS to learn an upper-level
    policy for gym CartPole environment.
"""

# Allow import of CREPS.py module in upper directory
import sys
import os.path
sys.path.append( os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

# Imports needed
import numpy as np
import gym
from CREPS      import computeSampleWeighting, UpperPolicy
from scenario   import LowerPolicy, predictReward # Scenario specific
from benchmarks import bench

# ------------------------------------------------------------------------------
# Contextual REPS algorithm parameters
# ------------------------------------------------------------------------------
eps = 1            # Relative entropy bound (lower -> more exploration)
M = 100            # Number of rollouts per policy iteration

# -----------------------------------------------------------------------------
# Scenario parameters
# -----------------------------------------------------------------------------
target = np.zeros(4) # Target state for policy
upper_a  = np.array([[1., 0., 1., 0.]]).T # Initial upper-policy parameters
upper_A = np.zeros((4, 4))
upper_sigma = np.eye(4) * [.1, .1, .1, .1]

# ------------------------------------------------------------------------------
# Initialization of necesary classes
# ------------------------------------------------------------------------------
pol = LowerPolicy(target)                           # Lower-policy
hpol = UpperPolicy(4)  # Upper-policy
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
while not solved:
    print('--------------------------------------')
    print('Run', k+1)
    k += 1

    R, W, F = predictReward(env, M, hpol, pol)
    p = computeSampleWeighting(R, F, eps)
    hpol.update(W, F, p)

    muR, solved = bench(env, hpol, pol, True)
