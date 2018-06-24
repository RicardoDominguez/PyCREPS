# Main script for the implementation of GPREPS

import numpy as np
from cost           import CostFcn
from GP             import GPS
from highpol        import HighPol
from policy         import TilePol
from predictReward  import predictReward
from plant          import Plant

# Indexes for the state variables being fed to GP, policy, cost function
dyni = [1, 3]      # GP inputs
dyno = [0, 1, 2]   # GP outputs
difi = [0, 1, 2]   # Variables trained by differences
ipol = [0]         # Policy inputs
icos = [2]         # Cost function inputs

# Algorithm parameters
eps = 3            # Relative entropy bound (lower -> more exploration)
K = 3              # Number of policy iterations
M = 10000          # Number of simulated rollouts
NinitRolls = 1     # Number of initial rollouts

# Simulated episode parameters
x0 = [0, 0, 0] # Initial state
H = 50 # Simulation horizon

# Low level policy
nX = H
deltaX = 1
minU = 0
maxU = 4200
pol = TilePol(nX, deltaX, minU, maxU)

# High level policy
W_mean = 2700
W_dev = W_mean / 10
hpol_mu = np.random.normal(W_mean, W_dev, nX);
hpol_sigma = np.eye(nX) * 4000
hpol = HighPol(hpol_mu, hpol_sigma)

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
for j in xrange(NinitRolls):
    x, y = sys.rollout(pol)
    X = np.append(X, x)
    Y = np.append(Y, y)

# Policy iteration
for k in xrange(K):
    gp.fit(X, Y)
    R, W, F = predictReward(x0, M, H, hpol, pol, gp, cost)
    hpol.update(R, W, F, eps)
    x, y = sys.rollout(pol)
    X = np.append(X, x)
    Y = np.append(Y, y)
