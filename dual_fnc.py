# GPREPS dual function minimizer
import numpy as np
import math
#from pyipm import IPM
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.misc import logsumexp
import pdb

def fnc(x, W, R, F, eps):
    '''
    Implementation of the GPREPS dual function

    Inputs:
        x           array containing (eta, theta)       (1 x 1+S)
        args        tuple containing the following arguments
            w       Weight  dataset matrix  (N x W)
            R       Return  dataset vector  (N x 1)
            F       Feature dataset matrix  (N x S)
            eps     Epsilon                 (1 x 1)

    Outputs:
        fval    value of the dual function at x         (1 x 1)
    '''
    eta, theta = x[0], np.array(x[1:]).reshape(-1, 1)

    # Evaluate dual function at x
    N = W.shape[0]
    mean_f = np.mean(F, 0).reshape(-1, 1)                  # (S x 1)
    err = R - np.sum(theta.T * F, 1).reshape(-1, 1)   # (N x 1)
    # zeta = np.exp(err / float(eta))                             # (N x 1)
    # s_zeta = np.sum(zeta)
    # summa = s_zeta / float(N)
    # fval = (eta * np.log(summa)) + (eta * eps) + (np.dot(theta.T, mean_f))
    log_summa = logsumexp(err / eta, b = 1.0 / N)
    fval = (eta * log_summa) + (eta * eps) + (np.dot(theta.T, mean_f))
    return fval

def fnc_der(x, W, R, F, eps):
    '''
    Implementation of the derivative of the GPREPS dual function

    Inputs:
        x           array containing (eta, theta)       (1 x 1+S)
        args        tuple containing the following arguments
            w       Weight  dataset matrix  (N x W)
            R       Return  dataset vector  (N x 1)
            F       Feature dataset matrix  (N x S)
            eps     Epsilon                 (1 x 1)

    Outputs:
        grad    gradient of the dual function at x      (1 x 1+S)
    '''
    eta, theta = x[0], np.array(x[1:]).reshape(-1, 1)

    N = W.shape[0]
    mean_f = np.mean(F, 0).reshape(-1, 1)                  # (S x 1)

    err = R - np.sum(theta.T * F, 1).reshape(-1, 1)   # (N x 1)
    # zeta = np.exp(err / float(eta))                             # (N x 1)
    # s_zeta = np.sum(zeta)
    # summa = s_zeta / float(N)
    log_summa = logsumexp(err / eta, b = 1.0 / N)

    # base = np.sum(zeta * err) / (eta * np.sum(zeta)) # (1 x 1)
    base = np.exp(logsumexp(err / eta, b = err) - logsumexp(err / eta) - np.log(eta))
    d_eta = eps + log_summa - base

    #pdb.set_trace()
    #d_theta = mean_f - (np.sum(zeta*F, 0).reshape(-1,1) / float(s_zeta)) # (S x 1)
    d_theta = np.zeros(mean_f.shape)

    grad = np.append(d_eta, d_theta).reshape(-1, 1) # (1 x S + 1)
    return grad

class DualFnc:
    def minimize(self, W, R, F, eps):
        '''
        Minimize dual function with constrain using the interior point algorithm.
        '''
        n_theta = F.shape[1]

        # Ensure initial point is defined
        #min_eta = np.max(R) / np.log(1e300)
        min_eta = 1e-299
        eta0 = 1
        # eta_undefined = True
        # eta0, theta0 = 1e-12, np.zeros((1, n_theta))
        # x0 = np.append(eta0, theta0)
        # #print 'check undef'
        # while(eta_undefined):
        #     eta0 = 10 * eta0
        #     x0 = np.append(eta0, theta0)
        #     pdb.set_trace()
        #     val = fnc(x0, W, R, F, eps)
        #     #print eta0
        #     #print val
        #     if eta0 > 1e50:
        #         break
        #     eta_undefined = math.isinf(val) or math.isnan(val)
        theta0 = np.zeros((1, n_theta))
        x0 = np.append(eta0, theta0)
        args = (W, R, F, eps)
        eta_bds = (min_eta, None)
        theta_bds = tuple([(None, None) for i in xrange(n_theta)])
        bds = tuple([eta_bds]) + theta_bds
        #problem = pyipm.IPM(x0=x0, x_dev=x_dev, f=f, ce=ce, ci=ci)
        #x = problem.solve(x0=x0)

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


# def fnc(self, x, eval_grad = False):
#     '''
#     Implementation of the GPREPS dual function
#
#     Inputs:
#         x           tuple containing (eta, theta)       (1 x 1+S)
#         eval_grad   if True the graident is evaluated
#
#     Outputs:
#         fval    value of the dual function at x         (1 x 1)
#         grad    gradient of the dual function at x      (1 x 1+S)
#     '''
#     eta, theta = x[0], np.array(x[1:]).reshape(-1, 1)
#
#     # Evaluate dual function at x
#     N = self.W.shape[0]
#     mean_f = np.mean(self.F, 0).reshape(-1, 1)                  # (S x 1)
#     err = self.R - np.sum(theta.T * self.F, 1).reshape(-1, 1)   # (N x 1)
#     zeta = np.exp(err / float(eta))                             # (N x 1)
#     s_zeta = np.sum(zeta)
#     summa = s_zeta / float(N)
#     fval = (eta * np.log(summa)) + (eta * self.eps) + (np.dot(theta.T, mean_f))
#
#     # Evaluate gradient
#     if eval_grad:
#         # Evaluate gradient w.r.t eta
#         base = np.sum(zeta * err) / (eta * np.sum(zeta)) # (1 x 1)
#         d_eta = self.eps + np.log(summa) - base
#
#         # Evaluate gradient w.r.t. theta
#         d_theta = mean_f - (np.sum(zeta * self.F, 0).reshape(-1,1) / float(s_zeta)) # (S x 1)
#         grad = np.append(d_eta, d_theta).reshape(1, -1) # (1 x S + 1)
#         return fval, grad
#     return fval
