'''
Script to compare the benefits of automatic gradients with theano.
'''

import numpy as np
import theano
import theano.tensor as T

import time

n_samples = 100
n_context_features = 10
Nevals = 1000

def compileDualFunction(R_, F_, eps_):
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

def compileDualFunction_hard(R_, F_, eps_):
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

    f = eta * (eps + log_sum_exp) + F_mean.dot(theta)
    d_eta = eps + log_sum_exp - (Z.dot(R_over_eta) / Z_sum)
    d_theta = F_mean - (Z.dot(F) / Z_sum)

    return theano.function([x], [f, d_eta, d_theta])

R = np.random.rand(n_samples, 1)
F = np.random.rand(n_samples, n_context_features)
eps = 1

# 1 Automatic diff
s = time.time()
t_compileDualFunction = compileDualFunction(R, F, eps)
for i in range(Nevals):
    x = np.random.rand(n_context_features + 1)
    a, b = t_compileDualFunction(x)
e = time.time() - s
print('1. Elapsed time:', e)

# 2 Hard coded diff
s = time.time()
t_compileDualFunction = compileDualFunction_hard(R, F, eps)
for i in range(Nevals):
    x = np.random.rand(n_context_features + 1)
    a, b, c = t_compileDualFunction(x)
    np.append(b, c)
e = time.time() - s
print('2. Elapsed time:', e)
