import numpy as np
import pdb
from cntxt import sampleContext

def predictReward(mod, M, H, hipol):
    F = sampleContext(M)
    W = hipol.contextSample(S = F, N = M).T
    R = mod.simulateRobot(M, H, F, W.T)
    return R.reshape(-1), W, F
