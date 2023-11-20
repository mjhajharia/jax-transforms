import jax.numpy as jnp
import jax
from tensorflow_probability.substrates import jax as tfp
import jax.scipy.special as jss
from functools import partial
import matplotlib.pyplot as plt
from jax.config import config
import numpy as np
import json
config.update("jax_enable_x64", True)

tfd = tfp.distributions
tfb = tfp.bijectors
tfpk = tfp.math.psd_kernels

import scipy.special as ss


import bridgestan as bs
import os
bs.set_bridgestan_path('/mnt/home/mjhajaria/.bridgestan/bridgestan-2.0.0')

@partial(jax.jit, static_argnames=['N'])
def ALR_stan_transform(y, N):
    log_det_jacobian = jnp.sum(jnp.sum(y) - N*jss.logsumexp(jnp.insert(y, 0, 0)) + 0.5*jnp.log(N))
    L = jax.nn.softmax(jnp.append(y,values=0))
    return L, log_det_jacobian

@partial(jax.jit, static_argnames=['N'])
def ALR_lp_stan(y, alpha, N):
    L, log_det_jacobian = ALR_stan_transform(y, N)
    return log_det_jacobian + tfd.Dirichlet(alpha.astype(jnp.float64)).log_prob(L)

@partial(jax.jit, static_argnames=['N'])
def Stickbreaking_stan_transform(y, N):
    x = jnp.empty(N)
    z = jss.expit(y - jnp.log(jnp.linspace(N-1, 1, N-1)))
    x = x.at[0].set(z[0])
    cum_sum = 0
    for i in range(1,N-1):
        cum_sum += x[i-1]
        x = x.at[i].set((1-cum_sum)*z[i])
    x.at[N-1].set(1 - (cum_sum+x[N-2]))
    log_det_jacobian = jnp.sum(jnp.log(z) + jnp.log1p(-z) + jnp.log1p(-jnp.cumsum(jnp.insert(x[:-2], 0, 0))))
    return x, log_det_jacobian

@partial(jax.jit, static_argnames=['N'])
def Stickbreaking_lp_stan(y, alpha, N):
    x, log_det_jacobian = Stickbreaking_stan_transform(y, N)
    return log_det_jacobian + tfd.Dirichlet(alpha.astype(jnp.float64)).log_prob(x)

@partial(jax.jit, static_argnames=['N'])
def HypersphericalProbit_stan_transform(y, N):
    x = jnp.empty(N)
    log_det_jacobian = -jax.lax.lgamma(float(N))
    sum_log_z = 0
    for i in range(N-1):
        log_u = tfd.Normal(0,1).log_cdf(y[i])
        log_z = log_u / (N-i-1)
        x.at[i].set(jnp.exp(sum_log_z + jnp.log(-jnp.expm1(log_z))))
        sum_log_z += log_z
        log_det_jacobian += tfd.Normal(0,1).log_prob(y[i])
    x.at[N-1].set(jnp.exp(sum_log_z))
    return x, log_det_jacobian

@partial(jax.jit, static_argnames=['N'])
def HypersphericalProbit_lp_stan(y, alpha, N):
    x, log_det_jacobian = HypersphericalProbit_stan_transform(y, N)
    return log_det_jacobian + tfd.Dirichlet(alpha.astype(jnp.float64)).log_prob(x)

@partial(jax.jit, static_argnames=['N'])
def HypersphericalLogit_stan_transform(y, N):
    x = jnp.empty(N)
    log_det_jacobian = -jax.lax.lgamma(float(N))
    sum_log_z = 0
    for i in range(N-1):
        log_u = jnp.log(jss.expit(y[i]))
        log_z = log_u / (N-i-1)
        x.at[i].set(jnp.exp(sum_log_z + jnp.log(-jnp.expm1(log_z))))
        sum_log_z += log_z
        log_det_jacobian += log_u + jnp.log(-jnp.expm1(log_u))
    x.at[N-1].set(jnp.exp(sum_log_z))
    return x, log_det_jacobian

@partial(jax.jit, static_argnames=['N'])
def HypersphericalLogit_lp_stan(y, alpha, N):
    x, log_det_jacobian = HypersphericalLogit_stan_transform(y, N)
    return log_det_jacobian + tfd.Dirichlet(alpha.astype(jnp.float64)).log_prob(x)

@partial(jax.jit, static_argnames=['N'])
def HypersphericalAngular_stan_transform(y, N):
    x = jnp.empty(N)
    log_det_jacobian = float(N-1)*jnp.log(float(2))
    s2_prod=float(1)
    log_halfpi = jnp.log(jnp.pi) - jnp.log(float(2))
    rcounter = int(2*N - 3)
    for i in range(N-1):
        z = jnp.log(jss.expit(y[i]))
        log_phi = z + log_halfpi
        phi = jnp.exp(log_phi)
        s = jnp.sin(phi)
        c = jnp.cos(phi)
        x.at[i].set(s2_prod*jnp.power(c, 2))
        s2_prod *= jnp.power(s,2)
        log_det_jacobian += log_phi+jnp.log(-jnp.expm1(z))+rcounter*jnp.log(s)+jnp.log(c)
        rcounter -= 2
    x.at[N-1].set(s2_prod)
    return x, log_det_jacobian

@partial(jax.jit, static_argnames=['N'])
def HypersphericalAngular_lp_stan(y, alpha, N):
    x, log_det_jacobian = HypersphericalAngular_stan_transform(y, N)
    return log_det_jacobian + tfd.Dirichlet(alpha.astype(jnp.float64)).log_prob(x)

@partial(jax.jit, static_argnames=['N'])
def AugmentedSoftmax_stan_transform(y, N):
    logr = jss.logsumexp(y)
    x = jnp.exp(y - logr)
    log_det_jacobian = jnp.sum(jnp.sum(y) - N*logr)
    log_det_jacobian += jnp.sum(tfd.Normal(0,1).log_prob(logr - jnp.log(N)))
    return x, log_det_jacobian

@partial(jax.jit, static_argnames=['N'])
def AugmentedSoftmax_lp_stan(y, alpha, N):
    x, log_det_jacobian = AugmentedSoftmax_stan_transform(y, N)
    return log_det_jacobian + tfd.Dirichlet(alpha.astype(jnp.float64)).log_prob(x)

@partial(jax.jit, static_argnames=['N'])
def ProbitProduct_stan_transform(y, N):
    zprod=1
    x = jnp.empty(N)
    log_det_jacobian = 0
    for i in range(N-1):
        yi = y[i]
        zi = tfd.Normal(0,1).cdf(yi)
        log_det_jacobian += jnp.sum(tfd.Normal(0,1).log_prob(yi))
        zprod_new = zprod*zi
        log_det_jacobian += jnp.sum(jnp.log(zprod))
        x.at[i].set(zprod - zprod_new)
        zprod = zprod_new
    x.at[N-1].set(zprod)
    return x, log_det_jacobian

@partial(jax.jit, static_argnames=['N'])
def ProbitProduct_lp_stan(y, alpha, N):
    x, log_det_jacobian = ProbitProduct_stan_transform(y, N)
    return log_det_jacobian + tfd.Dirichlet(alpha.astype(jnp.float64)).log_prob(x)

@partial(jax.jit, static_argnames=['N'])
def NormalizedExponential_stan_transform(y, N):
    r=0
    x = jnp.empty(N)
    z = jnp.empty(N)
    log_det_jacobian = 0
    for i in range(N-1):
        log_u = tfd.Normal(0,1).log_cdf(y[i])
        z.at[i].set(-jnp.log(-jnp.expm1(log_u)))
        r += z[i]
        log_det_jacobian += jnp.sum(tfd.Normal(0,1).log_prob(y[i]))
    x = z/r
    return x, log_det_jacobian

@partial(jax.jit, static_argnames=['N'])
def NormalizedExponential_lp_stan(y, alpha, N):
    x, log_det_jacobian = NormalizedExponential_stan_transform(y, N)
    return log_det_jacobian + tfd.Dirichlet(alpha.astype(jnp.float64)).log_prob(x)

def helmert_coding(N):
    if N<2:
        return("input must be >=2")
    Dm1 = N-1
    neg_ones = jnp.repeat(-1.0, Dm1)
    helmert_mat = jnp.vstack((jnp.expand_dims(neg_ones, axis=0), jnp.diag(jnp.arange(1, Dm1 + 1))))

    for i in range(1, N):
        for j in range(i, Dm1):
            helmert_mat = helmert_mat.at[i,j].set(-1.0)
    
    return helmert_mat

def make_v_fullrank(helmert_mat):
    N, Dm1 = helmert_mat.shape
    final_row = jnp.empty(N)
    
    if N-1 != Dm1:
        return("Matrix input must be size (D)x(D-1)")
    
    V = jnp.empty((Dm1, N))
    
    for i in range(Dm1):
        V = V.at[i,:].set(helmert_mat[:,i]/jnp.linalg.norm(helmert_mat[:,i]))
        final_row= final_row.at[i].set(0)
        
    final_row=final_row.at[N-1].set(1)
    return jnp.vstack((V, final_row))

def make_vinv(v):
    N, Dcol = v.shape
    if N!= Dcol:
        return("Rows and columns of input matrix must be equal")
    return jnp.linalg.inv(v)[:N-1, :N-1]

def construct_vinv(N):
    return make_vinv(make_v_fullrank(helmert_coding(N)))

@partial(jax.jit, static_argnames=['N'])
def AugmentedILR_stan_transform(y, N):
    Vinv = construct_vinv(N)
    s = jnp.insert(jnp.matmul(Vinv, y), N-1, 0)
    logr = jss.logsumexp(s)
    x = jnp.exp(s-logr)
    log_det_jacobian = jnp.sum(jnp.sum(s) - N*logr + jnp.log(N))
    return x, log_det_jacobian

@partial(jax.jit, static_argnames=['N'])
def AugmentedILR_lp_stan(y, alpha, N):
    x, log_det_jacobian = AugmentedILR_stan_transform(y, N)
    return log_det_jacobian + tfd.Dirichlet(alpha.astype(jnp.float64)).log_prob(x)