'''
Script to compare the benefits of using shared variables with Theano.

Compares the following scenarios:
    1. Passing all variables as arguments at every function call
    2. Using shared variables and recompiling when variables change
'''

import numpy as np
import theano
import theano.tensor as T

import time


n_samples = 100
n_context_features = 10
Nevals = 1000
Nchanges = 5

def compileDualFunction():
    x = T.dvector('x')
    R = T.dmatrix('R')
    F = T.dmatrix('F')
    eps = T.dscalar('eps')

    eta = x[0]
    theta = x[1:].reshape((-1, 1))

    F_mean = F.mean(0).reshape((1, -1))
    R_over_eta = (R - F.dot(theta)) / eta
    R_over_eta_max = R_over_eta.max()
    Z = T.exp(R_over_eta - R_over_eta_max).T
    Z_sum = Z.sum()
    log_sum_exp = R_over_eta_max + T.log(Z_sum / F.shape[0])

    # f wrapped in mean to prevent "cost must be a scalar" error
    f = T.mean(eta * (eps + log_sum_exp) + F_mean.dot(theta))
    d_x = T.grad(f, x)

    return theano.function([x, R, F, eps], [f, d_x])

def shared_compileDualFunction(R_, F_, eps_):
    x = T.dvector('x')
    
    R = theano.shared(R_)
    F = theano.shared(F_)
    eps = theano.shared(eps_)

    eta = x[0]
    theta = x[1:].reshape((-1, 1))

    F_mean = F.mean(0).reshape((1, -1))
    R_over_eta = (R - F.dot(theta)) / eta
    R_over_eta_max = R_over_eta.max()
    Z = T.exp(R_over_eta - R_over_eta_max).T
    Z_sum = Z.sum()
    log_sum_exp = R_over_eta_max + T.log(Z_sum / F.shape[0])

    # f wrapped in mean to prevent "cost must be a scalar" error
    f = T.mean(eta * (eps + log_sum_exp) + F_mean.dot(theta))
    d_x = T.grad(f, x)

    return theano.function([x], [f, d_x])

eps = 1

# 1 Single compilation
s = time.time()
t_compileDualFunction = compileDualFunction()
for c in range(Nchanges):
    R = np.random.rand(n_samples, 1)
    F = np.random.rand(n_samples, n_context_features)
    for i in range(Nevals):
        x = np.random.rand(n_context_features + 1)
        t_compileDualFunction(x, R, F, eps)
e = time.time() - s
print('1. Elapsed time:', e)

# 2 Shared with multiple compilation
s = time.time()
for c in range(Nchanges):
    R = np.random.rand(n_samples, 1)
    F = np.random.rand(n_samples, n_context_features)
    t_compileDualFunction = shared_compileDualFunction(R, F, eps)
    for i in range(Nevals):
        x = np.random.rand(n_context_features + 1)
        t_compileDualFunction(x)
e = time.time() - s
print('2. Elapsed time:', e)