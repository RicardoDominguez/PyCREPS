# GPREPS dual function minimizer
import numpy as np
from scipy.misc import logsumexp
from scipy.optimize import fmin_l_bfgs_b
import pdb

def computeSampleWeighting(W, R, F, eps):
    '''
    Inputs:
        w       Weight  dataset matrix  (N x W)
        R       Return  dataset vector  (N ,  )
        F       Feature dataset matrix  (N x S)
        eps     Epsilon                 (1 x 1)

    Outputs:
        p       policy update weights   (N x 1)
    '''
    # ----------------------------------------------------------------------
    # Minimize dual function using L-BFGS-B
    # ----------------------------------------------------------------------
    def dual_fnc(x): # Dual function with analyitical gradients
        eta = x[0]
        theta = x[1:]

        F_mean = F.mean(0)
        R_over_eta = R - theta.dot(F.T) / eta
        log_sum_exp = logsumexp(R_over_eta, b = 1.0 / W.shape[0])
        Z = np.exp(R_over_eta - R_over_eta.max())
        Z_sum = Z.sum()

        f = eta * (eps + log_sum_exp) + theta.T.dot(F_mean)
        d_eta = eps + log_sum_exp - (Z.dot(R_over_eta) / Z_sum)
        d_theta = F_mean - (Z.dot(F) / Z_sum)

        return f, np.append(d_eta, d_theta)

    # Initial point
    x0 = [1] + [1] * F.shape[1]

    # Bounds
    min_eta = 1e-10
    bds = np.vstack(([[min_eta, None]], np.tile(None, (F.shape[1], 2))))

    # Minimize using L-BFGS-B algorithm
    x = fmin_l_bfgs_b(dual_fnc, x0, approx_grad=None, bounds=bds)[0]

    # ----------------------------------------------------------------------
    # Determine weights of individual samples for policy update
    # ----------------------------------------------------------------------
    eta = x[0]
    theta = x[1:]

    R_baseline_eta = (R - theta.dot(F.T)) / eta
    p = np.exp(R_baseline_eta - R_baseline_eta.max())
    p /= p.sum()

    return p
