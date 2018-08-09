import numpy as np
from cost           import CostExpQuad
from highpol        import HighPol
from policy         import PID
from predictReward  import predictReward
import matplotlib.pyplot as plt
import pdb
from simulator import compareWeights
from simulator import Scenario
from simulator import validatePolicy
from dual_fnc import computeSampleWeighting

from robot_model import Model
from cntxt import sampleContext

# Algorithm parameters
eps = 1            # Relative entropy bound (lower -> more exploration)
K = 20             # Number of policy iterations
M = 1000           # Number of simulated policies
N = 1              # Number of rollouts per policy
NinitRolls = 1     # Number of initial rollouts

# Episode parameters
H = 300 # Horizon
dt = 0.02
scn = Scenario(dt, noise = False)

# Low level policy
target = np.array([10, 0])
offset = np.array([150, 150])
pol = PID(-324, 324, target, offset, maxI = 30, minI = -30, dt = dt)

# High level policy
hpol_mu =  np.array([-2, 100, 2, -100,0,0,0,0,0,0,0,0]).reshape(-1)
hpol_sigma = np.eye(hpol_mu.shape[0]) * [20, 200, 200, 20,0,0,0,0,0,0,0,0]
hpol = HighPol(hpol_mu, hpol_sigma, ns = 2)

# Cost function
Kcost = np.array([0.005, 100]).reshape(1, -1)
target = np.array([10, 0]).reshape(1, -1)
cost = CostExpQuad(Kcost, target)

# Rollout
mod = Model(dt, pol, cost, noise = False)

validatePolicy(scn, 100, H, pol, hpol, verbose = 0)

for k in xrange(K):
    print '--------------------------------------'
    print 'Run', k+1, 'out of', K
    R, W, F = predictReward(mod, M, H, hpol)
    p = computeSampleWeighting(W, R, F, eps)
    hpol.update(W, F, p)

validatePolicy(scn, 100, H, pol, hpol, verbose = 0)
