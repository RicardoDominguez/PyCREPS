import numpy as np

def bench(env, hipol, pol, verbose = False):
    '''
    Return the average reward over 100 episodes and whether the problem is solved.
    The problem is solved if all rewards are 200 (as specified by gym)
    '''
    N = 100
    R = np.zeros(N)
    for rollout in range(N):
        x = env.reset()                      # Sample context
        F = x[-2:]
        w = hipol.mean(F.reshape(1, -1)).T   # Sample lower-policy weights

        done = False
        while not done:
            u = pol.sample(w, x)
            x, r, done, info = env.step(u)
            R[rollout] += r

    muR = R.mean()
    stdR = np.std(R)

    if verbose:
        print('Mean reward: %.2f, std: %.2f' % (muR, stdR))

    return muR, stdR
