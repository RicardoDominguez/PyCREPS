# Main script for the implementation of GPREPS

import numpy as np
from cost import CostFcn
from GP import GPS
from highpol import HighPol
from policy import TilePol
from predictReward import predictReward
from plant import Plant

# Indexes for the state variables being fed to GP, policy, cost function
dyni = [2, 4]      # GP inputs
dyno = [1, 2, 3]   # GP outputs
difi = [1, 2, 3]   # Variables trained by differences
scal = [1, 1]      # Input scaling (before being fed to GP)
ipol = [1]         # Policy inputs
icos = [3]         # Cost function inputs

# Algorithm parameters
eps = 3            # Relative entropy bound (lower -> more exploration)
K = 3              # Number of policy iterations
M = 10000          # Number of simulated rollouts
NinitRolls = 1     # Number of initial rollouts
N = 20

# Simulated episode parameters
x0 = [0, 0, 0] # Initial state
H = 50 # Simulation horizon

# High level policy
hpol_mu = 2700;
hpol_sigma = np.eye(len(x0)) * 4000
hpol = HighPol(hpol_mu, hpol_sigma)

# Low level policy
nX = H
deltaX = np.linspace(0, H, H)
minU = 0
maxU = 4200
pol = TilePol(nX, deltaX, min, max)

# Cost function
K = 20
target = 0
cost = CostFcn(K, icos, target)

# Forward model
gp = GPS(dyni, dyno, difi)

# System being
sys = Plant()

# Initial rollouts
X, Y = np.empty([0, 0]), np.empty([0, 0])
for i in xrange(NinitRolls):
    x, y = sys.rollout(pol)
    X = np.append(X, x)
    Y = np.append(Y, y)

# Policy iteration
for j in xrange(M):
    gp.fit(X, Y)
    R, W, F = predictReward(x0, M, H, hpol, pol, gp, cost)
    hpol.update(R, W, F, eps)
    x, y = sys.rollout(pol)
    X = np.append(X, x)
    Y = np.append(Y, y)
