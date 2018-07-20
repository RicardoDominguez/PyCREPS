import numpy as np

class CostExpQuad:
    '''
    Compute the exponentiated negative quadratic cost:

        exp(-(x-z)^2 * w / 2)

    Where:
        x   state
        z   target state
        w   weight vector
    '''
    def __init__(self, w, z):
        '''
        Inputs
            w   weight vector   (1 x S)
            z   target state    (1 x S)
        '''
        self.w = w / 2.0
        self.z = z

    def sample(self, x):
        '''
        Inputs
            x   states       (N x S)

        Outputs
            C   cost         (N x 1)
        '''
        return np.exp(-np.sum(np.power(x - z, 2) * self.w, 1)).reshape(-1, 1)
