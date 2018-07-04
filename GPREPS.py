# Main script for the implementation of GPREPS

import numpy as np
from cost           import CostFcn
from GP             import GPS
from highpol        import HighPol
from policy         import TilePol
from predictReward  import predictReward
from plant          import Plant
import matplotlib.pyplot as plt

# Indexes for the state variables being fed to GP, policy, cost function
dyni = [1]      # GP inputs
dyno = [0]     # GP outputs
difi = [0]   # Variables trained by differences
ipol = [0]         # Policy inputs
icos = [0]         # Cost function inputs
nstates = 2

# Algorithm parameters
eps = 1            # Relative entropy bound (lower -> more exploration)
K = 10              # Number of policy iterations
M = 10000          # Number of simulated rollouts
NinitRolls = 1     # Number of initial rollouts

# Simulated episode parameters
x0 = np.array([0]) # Initial state
H = 10 # Simulation horizon

# Low level policy
nX = 10
deltaX = 2
minU = 0
maxU = 4
pol = TilePol(nX, deltaX, minU, maxU)

# High level policy
W_mean = 2
W_dev = W_mean / 1.0
hpol_mu = np.random.normal(W_mean, W_dev, nX);
print '-------------------------------------------'
print 'Initial policy weights', hpol_mu
hpol_sigma = np.eye(nX) * np.power(W_dev, 2)
hpol = HighPol(hpol_mu, hpol_sigma)

# Cost function
Kcost = 50
target = 0
cost = CostFcn(Kcost, icos, target)

# Forward model
gp = GPS(dyni, dyno, difi)

# System being
sys = Plant()

rewards = []

# Initial rollouts
X, Y = np.empty([0, nstates]), np.empty([0, nstates])
for j in xrange(NinitRolls):
    x, y = sys.rollout(hpol, pol)
    X = np.concatenate((X, x))
    Y = np.concatenate((Y, y))

# Policy iteration
for k in xrange(K):
    print '--------------------------------------'
    print 'Run', k+1, 'out of', K
    gp.fit(X, Y)
    R, W, F = predictReward(x0, M, H, hpol, pol, gp, cost)
    hpol.update(W, R, F, eps)
    x, y = sys.rollout(hpol, pol)
    X = np.concatenate((X, x))
    print 'Rollout y', y
    run_cost = cost.sample(y[:, 0].T)
    rewards.append(run_cost[0,0])
    print 'Rewards', rewards
    Y = np.concatenate((Y, y))
plt.plot(rewards)
plt.show()
