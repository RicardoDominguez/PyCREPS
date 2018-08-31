""" Main scrip illustrating the use of Contextual REPS to learn an upper-level
    policy for gym Acrobot environment.
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
import matplotlib.pyplot as plt

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
M = 500            # Number of rollouts per policy iteration
N = 15

# ------------------------------------------------------------------------------
# Scenario parameters
# ------------------------------------------------------------------------------
nS = 6
nF = 2
upper_a = np.zeros((1, 6)) #array([[0, 0, 0, ]])
upper_A = np.zeros((nF, nS))
upper_sigma = np.eye(nS) * [1, 1, 1, 1, 1, 1]

# ------------------------------------------------------------------------------
# Initialization of necesary classes
# ------------------------------------------------------------------------------
pol = LowerPolicy()
if use_torch:
    hpol = UpperPolicy(nF, torchOut = False)
else:
    hpol = UpperPolicy(nF, verbose=True)
hpol.set_parameters(upper_a, upper_A, upper_sigma)
env = gym.make('Acrobot-v1')

# ------------------------------------------------------------------------------
# Allow consistent results
# ------------------------------------------------------------------------------
# env.seed(10)
# np.random.seed(0)

# Benchmark of initial policy
print('--------------------------------------')
print('Initial policy...')
rewards = np.zeros((N + 1))
meanR, stdR = bench(env, hpol, pol, True)
rewards[0] = meanR

# ------------------------------------------------------------------------------
# Policy iteration
# ------------------------------------------------------------------------------
total_time = 0
for k in range(N):
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

    meanR, stdR = bench(env, hpol, pol, True)
    rewards[k] = meanR

print("Average update time", total_time/k)

plt.plot(rewards)
plt.show()
