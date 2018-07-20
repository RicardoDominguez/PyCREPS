# GPREPS dual function minimizer
import numpy as np
import math
import scipy.optimize as opt
from scipy.misc import logsumexp
import matplotlib.pyplot as plt

def fnc(x, W, R, F, eps):
    '''
    Implementation of the GPREPS dual function

    Inputs:
        x       (eta, theta)        (1 x 1+S)
        w       Weights             (N x W)
        R       Returns             (N x 1)
        F       Features            (N x S)
        eps     Epsilon             (1 x 1)

    Outputs:
        fval    value of the dual function at x         (1 x 1)
    '''
    eta = x[0]
    theta = np.array(x[1:]).reshape(-1, 1) # (S x 1)

    mean_f = np.mean(F, 0).reshape(-1, 1) # (S x 1)
    err = R - np.sum(theta.T * F, 1).reshape(-1, 1) # (N x 1)
    log_summa = logsumexp(err / eta, b = 1.0 / W.shape[0])

    fval = (eta * log_summa) + (eta * eps) + (np.dot(theta.T, mean_f))
    return fval

def fnc_der(x, W, R, F, eps):
    '''
    Computes the derivative of the GPREPS dual function

    Inputs:
        x       (eta, theta)        (1 x 1+S)
        w       Weights             (N x W)
        R       Returns             (N x 1)
        F       Features            (N x S)
        eps     Epsilon             (1 x 1)

    Outputs:
        grad    gradient of the dual function at x      (1 x 1+S)
    '''
    eta = x[0]
    theta = np.array(x[1:]).reshape(-1, 1) # (S x 1)

    mean_f = np.mean(F, 0).reshape(-1, 1) # (S x 1)
    err = R - np.sum(theta.T * F, 1).reshape(-1, 1) # (N x 1)
    log_summa = logsumexp(err / eta, b = 1.0 / W.shape[0])

    base = np.exp(logsumexp(err / eta, b = err) - logsumexp(err / eta) - np.log(eta)) # Change, its stupid
    d_eta = eps + log_summa - base

    #d_theta = mean_f - (np.sum(zeta*F, 0).reshape(-1,1) / float(s_zeta)) # (S x 1)
    d_theta = np.zeros(mean_f.shape) #

    grad = np.append(d_eta, d_theta).reshape(-1, 1) # (1 x S + 1)
    return grad

class DualFnc:
    def minimize(self, W, R, F, eps):
        '''
        Minimize the dual function using L-BFGS-B
        '''
        n_theta = F.shape[1]

        # Initial point
        eta0 = 1
        theta0 = np.zeros((1, n_theta))
        x0 = np.append(eta0, theta0)

        # Eta > 0
        min_eta = 1e-299 # Close to 0
        eta_bds = (min_eta, None)
        theta_bds = tuple([(None, None) for i in xrange(n_theta)])
        bds = tuple([eta_bds]) + theta_bds

        args = (W, R, F, eps)
        res = opt.minimize(fnc, x0, args, 'L-BFGS-B', fnc_der, bounds = bds, options={'disp': False})
        
        if res.success:
            return res.x
        else:
            print 'Maximum err', np.max(R)
            print res.message
            plt.plot(R)
            plt.show()
            raise Exception('Minimizer did not converge')

    def sampleWeighting(self, x, R, F):
        '''
        Compute the sample weighting for weighted ML update.
        '''
        eta = x[0]
        theta = x[1:]
        err = R - np.sum(theta.T * F, 1).reshape(-1, 1) # (N x 1)
        print 'Maximum err', np.max(err)
        print 'Eta', eta
        p = np.exp(err / eta)

        #plt.plot(p)
        #plt.show()
        return p # (N x 1)

    def computeSampleWeighting(self, W, R, F, eps):
        '''
        Inputs:
            w       Weight  dataset matrix  (N x W)
            R       Return  dataset vector  (N x 1)
            F       Feature dataset matrix  (N x S)
            eps     Epsilon                 (1 x 1)
        '''
        x = self.minimize(W, R, F, eps)
        return self.sampleWeighting(x, R, F)
