# GPREPS dual function minimizer

class DualFnc:
    def __init__(self):
        pass

    def updateArguments(self, W, R, F, eps):
        self.W = W
        self.R = R
        self.F = F
        self.eps = eps

    def updateEtaTheta(self, eta, theta):
        self.eta = eta
        self.theta = theta

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
        eta, theta = x[0], np.array(x[1:]).T

        # Evaluate dual function at x
        N = self.W.size
        mean_f = np.mean(self.F, 0).T                      # (S x 1)
        err = self.R - sum(np.dot(theta.T, self.F), 1)   # (N x 1)
        zeta = np.exp(err / float(eta))                      # (N x 1)
        s_zeta = np.sum(zeta)                                # (1 x 1)
        summa = s_zeta / float(N)                            # (1 x 1)
        fval = (eta * np.log(summa)) + (eta * self.eps) + (theta.T * mean_f) # (1 x 1)

        # Evaluate gradient
        if eval_grad:
            # Evaluate gradient w.r.t eta
            base = np.sum(zeta.dot(err)) / (eta * np.sum(zeta)) # (1 x 1)
            d_eta = self.eps + np.log(summa) - base

            # Evaluate gradient w.r.t. theta
            d_theta = mean_f - (np.sum(zeta.dot(self.F), 0).T / float(s_zeta)) # (S x 1)

            grad = np.array([d_eta, d_theta.T])             # (1 x S)
            return fval, grad
        return fval

    def minimize(self):
        '''
        Minimize dual function with constrain using the interior point algorithm.
        '''
        n_theta = self.F.shape[1]

        # Ensure initial point is defined
        eta_undefined = True
        eta0, theta0 = 1, np.zeros((1, n_theta))
        while(eta_undefined):
            eta0 = 10 * eta0
            x0 = np.array(eta0, theta0)
            val, grad = self.fcn(x0)
            eta_undefined = math.isinf(val) or math.isnan(val) or math.isinf(grad(1)) or math.isnan(grad(1))

        X = minimize() # TODO
        self.updateEtaTheta() #TODO

    def sampleWeighting(self):
        err = self.R - np.sum(np.dot(self.theta.T, self.F), 1)
        p = np.exp(err / self.eta)
        return p

    def computeSampleWeighting(self, W, R, F, eps):
        self.updateArguments(W, R, F, eps)
        self.minimize()
        return self.sampleWeighting()
