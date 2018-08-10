""" Main scrip illustrating the use of Contextual REPS to learn an upper-level
    policy for the particular scenario/environment of a wall folliwng robot.
"""

# Allow import of CREPS.py module in upper directory
import sys
import os.path
sys.path.append( os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

# Imports needed
import numpy as np
from CREPS    import computeSampleWeighting, UpperPolicy
from scenario import Cost, LowerPolicy, Model, predictReward # Scenario specific
from benchmarks import validatePolicy

# ------------------------------------------------------------------------------
# Contextual REPS algorithm parameters
# ------------------------------------------------------------------------------
eps = 1            # Relative entropy bound (lower -> more exploration)
K = 10             # Number of policy iterations
M = 1000           # Number of rollouts per policy iteration

# -----------------------------------------------------------------------------
# Scenario parameters
# -----------------------------------------------------------------------------
H = 300     # Episode step-horizon
dt = 0.1    # in seconds
target = np.array([10, 0]) # Target state for policy and cost function
upper_mean  =  np.array([-2, 100, 2, -100, 0, 0, 0, 0, 0, 0, 0, 0]) # Initial upper-policy parameters
upper_sigma = np.eye(upper_mean.shape[0]) * [20, 200, 200, 20, 0, 0, 0, 0, 0, 0, 0, 0]

# ------------------------------------------------------------------------------
# Initialization of necesary classes
# ------------------------------------------------------------------------------
offset  = np.array([150, 150])
pol     = LowerPolicy(-324, 324, target, offset, maxI = 30, minI = -30, dt = dt) # Lower-policy
hpol    = UpperPolicy(upper_mean, upper_sigma, nS = 2)                           # Upper-policy
cost    = Cost(np.array([0.005, 100]), target)                                   # Cost function
mod     = Model(dt, pol, cost, noise = False)                                    # Rollout model

# Benchmark of initial policy
validatePolicy(100, H, dt, pol, hpol, verbose = 0)

# ------------------------------------------------------------------------------
# Policy iteration
# ------------------------------------------------------------------------------
for k in xrange(K):
    print '--------------------------------------'
    print 'Run', k+1, 'out of', K

    R, W, F = predictReward(mod, M, H, hpol)
    p = computeSampleWeighting(R, F, eps)
    hpol.update(W, F, p)

# Benchmark of end policy
validatePolicy(100, H, dt, pol, hpol, verbose = 0)
