import numpy as np
from sklearn import gaussian_process

class GPS:
    '''
    Wrapper for GP dynamics model using sklearn
    '''
    def __init__( self, dyni, dyno, difi):
        self.dyni = dyni # GP inputs
        self.dyno = dyno # GP outputs
        self.difi = difi # Variables trained by differences
        self.nout = len(dyno)
        self.gps = [gaussian_process.GaussianProcessRegressor() for i in range(self.nout)]

    def fit(self, X, Y):
        '''
        Fit the dynamic model using rollout data.
        '''
        Yd = np.copy(Y)
        Xt = X[:, self.dyni] # Inputs of the model
        Yd[:, self.difi] = Y[:, self.difi] - X[:, self.difi] # Differences
        Yt = Yd[:, self.dyno] # Outputs of the model
        for i in xrange(self.nout):
            try:
                self.gps[i].fit(Xt, Yt[:, i])
            except ValueError as e:
                print( 'ValueError cought for i:{0}: e:{1}'.format( i, e ) )
                raise e

    def predict(self, X):
        '''
        Predict dynamics using the mean of the GP rather than sampling
        '''
        GPin = X[:, self.dyni]
        Y = np.empty((X.shape[0], self.nout))
        for i in xrange(self.nout):
            Y[:, i] = self.gps[i].predict(GPin)
        Y[:, self.difi] += X[:, self.difi]
        return Y
