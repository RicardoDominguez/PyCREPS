# Main script for the implementation of GPREPS

import numpy as np
from cost           import CostExpQuad
from highpol        import HighPol
from policy         import Proportional
from predictReward  import predictReward
from plant          import Plant
import matplotlib.pyplot as plt
import pdb

# Indexes for the state variables being fed to GP, policy, cost function
dyni = [1]      # GP inputs
dyno = [0]     # GP outputs
difi = [0]   # Variables trained by differences
ipol = [0]         # Policy inputs
icos = [0]         # Cost function inputs
nstates = 2

# Algorithm parameters
eps = 1            # Relative entropy bound (lower -> more exploration)
K = 10             # Number of policy iterations
M = 100           # Number of simulated rollouts
NinitRolls = 1     # Number of initial rollouts

# Simulated episode parameters
x0 = np.array([35, 0]) # Initial state
H = 1000 # Simulation horizon

# Low level policy
minU = -324
maxU = 324
target = np.array([20, 0]).reshape(-1)
offset = np.array([150, 150]).reshape(-1)
pol = Proportional(minU, maxU, target, offset)

# High level policy
hpol_mu = np.array([-0.5, 10, -0.5, -10]).reshape(-1)
hpol_sigma = np.eye(hpol_mu.shape[0]) * [10, 20, 10, 20]
hpol = HighPol(hpol_mu, hpol_sigma)
#pdb.set_trace()

# Cost function
Kcost = np.array([0.005, 100]).reshape(1, -1)
target = np.array([20, 0]).reshape(1, -1)
cost = CostExpQuad(Kcost, target)

# System being
sys = Plant()

rewards = []

# Policy iteration
X, Y = np.empty([0, nstates]), np.empty([0, nstates])
for k in xrange(K):
    print '--------------------------------------'
    print 'Run', k+1, 'out of', K
    R, W, F = predictReward(x0, M, H, hpol, pol, cost)
    hpol.update(W, R, F, eps)
    R = sys.rollout(x0, H, hpol, pol, cost)
    rewards.append(R)
    print 'Rewards', rewards

plt.plot(rewards)
plt.show()
