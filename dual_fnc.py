# GPREPS dual function minimizer
import numpy as np
from pyipm import IPM

class DualFnc:
    def updateArguments(self, W, R, F, eps):
        '''
        Inputs:
            w       Weight  dataset matrix  (N x W)
            R       Return  dataset vector  (N x 1)
            F       Feature dataset matrix  (N x S)
            eps     Epsilon                 (1 x 1)
        '''
        self.W = W
        self.R = R
        self.F = F
        self.eps = eps

    def updateEtaTheta(self, x):
        '''
        Changes:
            eta         (1 x 1)
            theta       (S x 1)
        '''
        self.eta = x[0]
        self.theta = theta[1:]

    def fnc(self, x, eval_grad = False):
        '''
        Implementation of the GPREPS dual function

        Inputs:
            x           tuple containing (eta, theta)       (1 x 1+S)
            eval_grad   if True the graident is evaluated

        Outputs:
            fval    value of the dual function at x         (1 x 1)
            grad    gradient of the dual function at x      (1 x 1+S)
        '''
        eta, theta = x[0], np.array(x[1:]).reshape(-1, 1)

        # Evaluate dual function at x
        N = self.W.shape[0]
        mean_f = np.mean(self.F, 0).reshape(-1, 1)                  # (S x 1)
        err = self.R - np.sum(theta.T * self.F, 1).reshape(-1, 1)   # (N x 1)
        zeta = np.exp(err / float(eta))                             # (N x 1)
        s_zeta = np.sum(zeta)
        summa = s_zeta / float(N)
        fval = (eta * np.log(summa)) + (eta * self.eps) + (np.dot(theta.T, mean_f))

        # Evaluate gradient
        if eval_grad:
            # Evaluate gradient w.r.t eta
            base = np.sum(zeta * err) / (eta * np.sum(zeta)) # (1 x 1)
            d_eta = self.eps + np.log(summa) - base

            # Evaluate gradient w.r.t. theta
            d_theta = mean_f - (np.sum(zeta * self.F, 0).reshape(-1,1) / float(s_zeta)) # (S x 1)
            grad = np.append(d_eta, d_theta).reshape(1, -1) # (1 x S + 1)
            return fval, grad
        return fval

    def minimize(self):
        '''
        Minimize dual function with constrain using the interior point algorithm.
        '''
        n_theta = self.F.shape[1]

        # Ensure initial point is defined
        #eta_undefined = True
        eta0, theta0 = 1, np.zeros((1, n_theta))
        x0 = np.append(eta0, theta0)
        # while(eta_undefined):
        #     eta0 = 10 * eta0
        #     x0 = np.append(eta0, theta0)
        #     val, grad = self.fcn(x0)
        #     eta_undefined = math.isinf(val) or math.isnan(val) or math.isinf(grad(1)) or math.isnan(grad(1))

        problem = pyipm.IPM(x0=x0, x_dev=x_dev, f=f, ce=ce, ci=ci)
        x = problem.solve(x0=x0)
        self.updateEtaTheta(x)

    def sampleWeighting(self):
        '''
        Compute the sample weighting for weighted ML update.
        '''
        err = self.R - np.sum(self.theta.T * self.F, 1).reshape(-1, 1) # (N x 1)
        return np.exp(err / self.eta) # (N x 1)

    def computeSampleWeighting(self, W, R, F, eps):
        '''
        Inputs:
            w       Weight  dataset matrix  (N x W)
            R       Return  dataset vector  (N x 1)
            F       Feature dataset matrix  (N x S)
            eps     Epsilon                 (1 x 1)
        '''
        self.updateArguments(W, R, F, eps)
        self.minimize()
        return self.sampleWeighting()
