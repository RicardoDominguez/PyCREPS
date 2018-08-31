import torch

def bench(env, hipol, pol, verbose = False):
    """Benchmark the upper-level policy

    Return the average reward over 100 episodes and whether the problem is solved.
    The problem is solved when the mean reward over 100 episodes is equal or
    greater than 195 (as specified by gym).

    Parameters
    ----------

    env: gym environment

    hipol: UpperPolicy

    pol: LowerPolicy

    verbose: bool, optional (default: False)
        Prints additional information
    """
    N = 100
    R = torch.zeros(N)
    for rollout in range(N):
        x = torch.from_numpy(env.reset()).view(1, -1)
        w = hipol.mean(x).t() # Sample lower-policy weights

        done = False
        while not done:
            u = pol.sample(w, x)
            x, r, done, info = env.step(u)
            x = torch.from_numpy(x).view(1, -1)
            R[rollout] += r

    muR = round(float(R.mean()), 3)
    solved = muR >= 195

    if verbose:
        print('Mean reward', muR)
        if solved:
            print('Solved!')
        else:
            print('Not solved.')

    return muR, solved
