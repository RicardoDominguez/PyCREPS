# Main script for the implementation of GPREPS

import numpy as np
from cost           import CostExpQuad
from highpol        import HighPol
from policy         import Proportional
from policy         import PropInt
from predictReward  import predictReward
from plant          import Plant
import matplotlib.pyplot as plt
import pdb
from simulator import compareWeights
from simulator import Scenario
from simulator import validatePolicy

from sim_opt import OptMod

# Good weights... [-31, 252, -24.6, -105.74]
# Better weights... [-19.3, 190.8, -13.23, -115.6]
# Trained for wider range... [-10.5, 222.85, -0.0116, -110.8]

# Indexes for the state variables being fed to GP, policy, cost function
dyni = [1]      # GP inputs
dyno = [0]      # GP outputs
difi = [0]      # Variables trained by differences
ipol = [0]      # Policy inputs
icos = [0]      # Cost function inputs
nstates = 2

# Algorithm parameters
eps = 1            # Relative entropy bound (lower -> more exploration)
K = 20             # Number of policy iterations
M = 100           # Number of simulated policies
N = 1            # Number of rollouts per policy
NinitRolls = 1     # Number of initial rollouts

# Simulated episode parameters
x0 = np.array([200, np.pi/4]) # Initial state
H = 1000 # Simulation horizon

# Low level policy
dt = 0.02
minU = -324
maxU = 324
target = np.array([10, 0]).reshape(-1)
offset = np.array([150, 150]).reshape(-1)
#pol = Proportional(minU, maxU, target, offset)
pol = PropInt(minU, maxU, target, offset, maxI = 20, minI = -20, dt = dt)

# High level policy
hpol_mu =  np.array([-2, 100, 2, -100, -0.001, 0.005, 0.001, -0.005]).reshape(-1)
hpol_sigma = np.eye(hpol_mu.shape[0]) * [20, 200, 200, 20, 5, 5, 5, 5]
hpol = HighPol(hpol_mu, hpol_sigma)
#pdb.set_trace()

# Cost function
Kcost = np.array([0.005, 100]).reshape(1, -1)
target = np.array([10, 0]).reshape(1, -1)
cost = CostExpQuad(Kcost, target)

# System being
scn = Scenario(dt, noise = False)
sys = Plant()
mod = OptMod(dt, pol, cost, noise = False)

rewards = []

# Policy iteration
X, Y = np.empty([0, nstates]), np.empty([0, nstates])
'Initial rollout....'
R = sys.rollout(scn, x0, H, hpol, pol, cost)
rewards.append(R)
for k in xrange(K):
    print '--------------------------------------'
    print 'Run', k+1, 'out of', K
    R, W, F = predictReward(mod, x0, M, N, H, hpol)
    hpol.update(W, R, F, eps)
    R = sys.rollout(scn, x0, H, hpol, pol, cost)
    rewards.append(R)
    print 'Rewards', rewards

plt.plot(rewards)
plt.show()

compareWeights(scn, x0, H, pol, hpol_mu, hpol.mu)

x0s = np.array([[200, np.pi/3], [200, np.pi/6], [200, np.pi/4], [200, np.pi/3.5], [200, np.pi/5]])
validatePolicy(scn, x0s, H, pol, hpol.mu)
