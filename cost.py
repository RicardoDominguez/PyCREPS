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
        #return np.exp(-np.sum(np.power(x - self.z, 2) * self.w, 1)).reshape(-1, 1)
        if x[0] > 0.1:
            return np.exp(-np.sum(np.abs(x - self.z) * self.w, 1)).reshape(-1, 1) / 10
        else:
            return np.array([[-1]]) / 10

    def sampleMat(self, x):
        '''
        Inputs
            x   states      (N x S)

        Outputs
            C   cost        (N x 1)
        '''
        C = np.empty((x.shape[0], 1))
        vald = x[:, 0] > 0.1
        C[vald] = np.exp(-np.sum(np.abs(x[vald, :] - self.z) * self.w, 1)).reshape(-1, 1)
        vald = np.invert(vald)
        C[vald] = -1
        return C / 10

if __name__ == '__main__':
    # w = np.array([0.005, 100]).reshape(1, -1)
    # t = np.array([20, 0]).reshape(1, -1)
    # cost = CostExpQuad(w, t)
    # print 'max ang ', cost.sample([0, np.pi/2])
    # print 'max tof ', cost.sample([255, 0])
    # print 'both max ', cost.sample([255, 1.57])
    # print 'reg tof', cost.sample([30, np.pi/4])
    # print 'reg ang', cost.sample([60, np.pi/16])
    # print 'ttt ang', cost.sample([60, 0])
    # print 'ttt tof', cost.sample([30, np.pi/16])
    w = np.array([0.5, 1]).reshape(1, -1)
    t = np.array([10, 0]).reshape(1, -1)
    cost = CostExpQuad(w, t)
    x = np.array([[10, 0], [20, 0], [1, 1]])
    print cost.sampleMat(x)
    print cost.sample(x[0, :])
    print cost.sample(x[1, :])
    print cost.sample(x[2, :])
    # print 'zero ', cost.sample([10, 20])
    # print 'at wall', cost.sample([0, 0])
    # print 'init ', cost.sample([30, 0])
    # print 'very far', cost.sample([50, 0])
