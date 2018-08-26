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
from CREPS_numpy    import computeSampleWeighting, UpperPolicy
from scenario       import LowerPolicy, predictReward, systemRollout
from GP             import GPS
from benchmarks     import bench

# ------------------------------------------------------------------------------
# Contextual REPS algorithm parameters
# ------------------------------------------------------------------------------
eps = 1            # Relative entropy bound (lower -> more exploration)
M = 1000           # Number of rollouts per policy iteration
NinitRolls = 1     # Number of initial rollouts

# ------------------------------------------------------------------------------
# Indexes for learning GP model, state (v, dv, theta, dtheta, u)
# ------------------------------------------------------------------------------
dyni = [1, 2, 3, 4]     # GP inputs
dyno = [0, 1, 2, 3]     # GP outputs
difi = [0, 1, 2, 3]     # Variables trained by differences
gp = GPS(dyni, dyno, difi)

# -----------------------------------------------------------------------------
# Scenario parameters
# -----------------------------------------------------------------------------
target = np.zeros(4) # Target state for policy
upper_a = np.array([[1, 0, 1, 0]]).T # Initial upper-policy parameters
upper_A = np.zeros((4, 4))
upper_sigma = np.eye(upper_a.shape[0]) * [.1, .1, .1, .1]

# ------------------------------------------------------------------------------
# Initialization of necesary classes
# ------------------------------------------------------------------------------
pol     = LowerPolicy(target) # Lower-policy
hpol    = UpperPolicy(4)  # Upper-policy
hpol.set_parameters(upper_a, upper_A, upper_sigma)
env     = gym.make('CartPole-v0')

# ------------------------------------------------------------------------------
# Allow consistent results
# ------------------------------------------------------------------------------
env.seed(10)
np.random.seed(4)

# Benchmark of initial policy
print('--------------------------------------')
print('Initial policy...')
muR, solved = bench(env, hpol, pol, True)

# ------------------------------------------------------------------------------
# Initial rollouts
# ------------------------------------------------------------------------------
X, Y = np.empty([0, 5]), np.empty([0, 5])
for j in range(NinitRolls):
    x, y = systemRollout(env, hpol, pol)
    X = np.concatenate((X, x))
    Y = np.concatenate((Y, y))

# ------------------------------------------------------------------------------
# Policy iteration
# ------------------------------------------------------------------------------
k = 1
while not solved:
    print('--------------------------------------')
    print('Run', k)
    k += 1

    print('Fitting GP model...')
    gp.fit(X, Y)

    print('Simulating rollouts...')
    R, W, F = predictReward(M, hpol, pol, gp)
    p = computeSampleWeighting(R, F, eps)
    hpol.update(W, F, p)

    x, y = systemRollout(env, hpol, pol)
    X = np.concatenate((X, x))
    Y = np.concatenate((Y, y))

    muR, solved = bench(env, hpol, pol, True)
