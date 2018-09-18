'''
Script to compare the benefits of automatic gradients with pytorch.
'''

import numpy as np
import torch

import time

n_samples = 1000
n_context_features = 100000
Nevals = 100

R = torch.from_numpy(np.random.rand(n_samples, 1))
F = torch.from_numpy(np.random.rand(n_samples, n_context_features))
eps = 1

def dual_fnc(x_): # Dual function with analyitical gradients
    x = torch.as_tensor(x_)
    eta = x[0]
    theta = x[1:].view(-1,1)

    F_mean = F.mean(0).view(1,-1)
    R_over_eta = (R - F.mm(theta)) / eta
    R_over_eta_max = R_over_eta.max()
    Z = torch.exp(R_over_eta - R_over_eta_max)
    Z_sum = Z.sum()
    log_sum_exp = R_over_eta_max + torch.log(Z_sum / F.shape[0])

    f = eta * (eps + log_sum_exp) + F_mean.mm(theta)        
    d_eta = eps + log_sum_exp - Z.t().mm(R_over_eta)/Z_sum
    d_theta = F_mean - (Z.t().mm(F) / Z_sum)
    
    return f.numpy(), np.append(d_eta.numpy(), d_theta.numpy())
    
def dual_fnc_auto(x_): # Dual function with analyitical gradients
    x = torch.as_tensor(x_)
    x.requires_grad_()
    
    eta = x[0]
    theta = x[1:].view(-1,1)

    F_mean = F.mean(0).view(1,-1)
    R_over_eta = (R - F.mm(theta)) / eta
    R_over_eta_max = R_over_eta.max()
    Z = torch.exp(R_over_eta - R_over_eta_max)
    Z_sum = Z.sum()
    log_sum_exp = R_over_eta_max + torch.log(Z_sum / F.shape[0])

    f = eta * (eps + log_sum_exp) + F_mean.mm(theta)
    f.backward()
    
    return f.data.numpy(), x.grad.numpy()

# 1 Hard coded diff
s = time.time()
for i in range(Nevals):
    x = np.random.rand(n_context_features + 1)
    dual_fnc(x)
e = time.time() - s
print('1. Elapsed time:', e)

# 2 Automatic diff
s = time.time()
for i in range(Nevals):
    x = np.random.rand(n_context_features + 1)
    dual_fnc_auto(x)
e = time.time() - s
print('2. Elapsed time:', e)