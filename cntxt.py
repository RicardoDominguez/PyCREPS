import numpy as np

def sampleContext(N):
    S = np.empty((N, 2))
    S[:, 0] =  np.random.rand(N) * 50 + 150
    S[:, 1] =  np.random.rand(N) * 0.5236 + 0.5236
    return S
