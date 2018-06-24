def predictReward(x0, M, H, hipol, pol, gp, cost):
    '''
    Compute expected rewards by sampling trayectories using the forward models.
    '''
    ns = x0.size

    # Sample low level policy weight from high level policy
    w = hipol.sample(M) # (W x M)

    # Initialize matrices
    x = np.dot(np.ones((M, 1)), x0) # Initial states
    y = np.zeros((M, ns, H))

    # For each step within horizon
    for t in xrange(H):
        u = pol.sample(w, x)
        xu = np.array([x, u]) # (N x nS + 1)
        gpout = gp.predict(xu)
        y[:, :, t] = gpout
        x = gpout

    R = cost.sample(y) # Episode reward
    W = w.T
    F = np.zeros(M, ns)

    return R, W, F
