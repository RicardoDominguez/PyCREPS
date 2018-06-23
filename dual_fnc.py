# GPREPS dual function minimizer

class Dual_fnc:
    def __init__(self):
        pass

    def fnc(self, x, eval_grad = 0):
        '''
        Implementation of the GPREPS dual function

        Inputs:
            x           tuple containing (eta, theta)       (1 x 1+S)
            eval_grad   if True the graident is evaluated

        Outputs:
            fval    value of the dual function at x         (1 x 1)
            grad    gradient of the dual function at x      (1 x 1+S)
        '''
        eta, theta = x[0], np.array(x[1:]).T

        # Extract addition arguments (TODO)
        Wdataset = # (N x W)
        Rdataset = # (N x 1)
        Fdataset = # (N x S)
        eps =      # (1 x 1)

        # Evaluate dual function at x
        N = Wdataset.size
        mean_f = np.mean(Fdataset, 0).T                      # (S x 1)
        err = Rdataset - sum(np.dot(theta.T, Fdataset), 1)   # (N x 1)
        zeta = np.exp(err / float(eta))                      # (N x 1)
        s_zeta = np.sum(zeta)                                # (1 x 1)
        summa = s_zeta / float(N)                            # (1 x 1)
        fval = (eta * np.log(summa)) + (eta * eps) + (theta.T * mean_f) # (1 x 1)

        # Evaluate gradient
        if eval_grad:
            # Evaluate gradient w.r.t eta
            base = np.sum(zeta.dot(err)) / (eta * np.sum(zeta)) # (1 x 1)
            d_eta = eps + np.log(summa) - base

            # Evaluate gradient w.r.t. theta
            d_theta = mean_f - (np.sum(zeta.dot(Fdataset), 0).T / float(s_zeta)) # (S x 1)
