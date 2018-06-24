import numpy as np
from sklearn import gaussian_process
from sklearn.base import BaseEstimator

MACHINE_EPSILON = np.finfo(np.double).eps

class GPS( BaseEstimator ):
    '''
    multivariete wrapper for gaussian process model for sklearn
    '''

    def __init__( self, dyni, dyno, difi, regr='constant', corr='squared_exponential',
                 storage_mode='full', verbose=False, theta0=1e-1 ):
        self.dyni = dyni
        self.dyno = dyno
        self.difi = difi
        self.nout = len(dyno)
        self.gps = [ gaussian_process.GaussianProcess( regr=regr, corr=corr,
                 storage_mode=storage_mode, verbose=verbose, theta0=theta0 ) for i in range( self.nout ) ]

    def fit(self, X, Y):
        Xt = X[:, self.dyni] # Inputs of the model
        Y[:, self.difi] -= X[:, self.difi] # Differences
        Yd = Y[:, self.dyno] # Outputs of the model

        for i in xrange(self.nout):
            try:
                self.gps[i].fit(Xt, Yd[:, i])
            except ValueError as e:
                print( 'ValueError cought for i:{0}: e:{1}'.format( i, e ) )
                raise e

    def predict(self, X):
        GPin = X[:, self.dyni]
        Y = np.empty((X.shape[0], self.nout))
        for i in xrange(self.nout):
            Y[:, i] = self.gps[i].predict(GPin)
        Y[:, self.difi] += X[:, self.difi]
        return Y
