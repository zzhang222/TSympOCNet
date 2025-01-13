from jax import random, nn, jit, jacrev, jvp
import jax.numpy as jnp
from functools import partial
from jax import vmap, grad
from utils import relu
from scipy import optimize
import numpy as np
from math import ceil

act_fun = 'swish'
if act_fun == 'tanh':
    act = jnp.tanh
elif act_fun == 'sigmoid':
    act = nn.sigmoid
elif act_fun == 'relu':
    act = relu
elif act_fun == 'swish':
    act = lambda x: x * nn.sigmoid(x)
elif act_fun == 'leaky_relu':
    act = lambda x: jnp.maximum(0.1*x, x)

def init_gradient_params(d, width, key, scale = 1e-2):
    K_key, a_key = random.split(key)
    return scale * random.normal(K_key, (width, d)), scale * random.normal(a_key, (width,)), jnp.zeros(width)

def init_sympnet_params(d, width, key, layers):
    keys = random.split(key, layers)
    return [init_gradient_params(d, width, key) for key in keys]

def init_pinn_params(d, key, width, layers):
    return [init_fnn_params(d, key, width, layers), init_fnn_params(d, key, width, layers)]

@partial(vmap, in_axes = (None, 0, 0, 0, None))
def identity(net_params, Q, P, t, t_terminal):
    return Q, P

@partial(vmap, in_axes = (None, 0, None))
def pinn(latent_params, t, t_terminal):
    new_t = t / t_terminal
    q_params, p_params = latent_params
    q = fnn(q_params, new_t, scale = 1)
    p = fnn(p_params, new_t, scale = 1)
    return q, p

def precompute_latent_fixed(bdy, t, t_terminal):
    p = (bdy[1] - bdy[0]) / t_terminal
    p = jnp.tile(p[None, :], (len(t), 1))
    q = vmap(lambda t, q: jnp.interp(t.flatten(), jnp.array([0, t_terminal]), q), in_axes = (None, 1), out_axes = 1)(t, bdy)
    return q, p

def get_latent_pre(latent_params, t, t_terminal, qp):
    q, p = qp
    t_grid = jnp.linspace(0, t_terminal, len(q))
    q_new = vmap(lambda t, q: jnp.interp(t.flatten(), t_grid, q), in_axes = (None, 1), out_axes = 1)(t, q)
    p_new = vmap(lambda t, p: jnp.interp(t.flatten(), t_grid, p), in_axes = (None, 1), out_axes = 1)(t, p)
    return q_new, p_new

def jvp_pre(transform, get_latent, net_params, latent_params, t, t_terminal, *args):
    Q, P = get_latent(latent_params, t, t_terminal, *args)
    dQ = P
    dP = jnp.zeros_like(P)
    transform_partial = lambda Q, P, t: transform(net_params, Q, P, t, t_terminal)
    return jvp(transform_partial, (Q, P, t), (dQ, dP, jnp.ones_like(t)))

def jvp_pinn(transform, get_latent, net_params, latent_params, t, t_terminal, *args):
    transform_partial = lambda t: get_latent(latent_params, t, t_terminal, *args)
    return jvp(transform_partial, (t,), (jnp.ones_like(t),))

def random_layer_params(d, width, key, scale = 1e-2):
    K_key, a_key = random.split(key)
    return scale * random.normal(K_key, (width, d)), scale * random.normal(a_key, (width,)), jnp.zeros(width)

def init_network_params(d, width, key, layers):
    keys = random.split(key, layers)
    return [random_layer_params(d, width, key) for key in keys]

def init_latent_params(d):
    return jnp.ones((1, d)), jnp.ones((1, d)), jnp.ones((1, d))

def init_sim_params(d, width, key, scale = 1e-2):
    return scale * random.normal(key, (width, d)), jnp.zeros(width)

def init_tsympnet_params(d, width, subwidth, key, layers, sublayers):
    sim_key, diag_key = random.split(key)
    sim_keys, diag_keys = random.split(sim_key, layers), random.split(diag_key, layers)
    return [init_sim_params(d, width, key) for key in sim_keys], [init_fnn_params(width, key, subwidth, sublayers) for key in diag_keys]

def fnn_layer_params(ind, outd, key):
    scale = jnp.sqrt(2 / (ind + outd))
    return random.normal(key, (ind, outd)) * scale, jnp.zeros(ind)

def init_fnn_params(d, key, width, layers):
    keys = random.split(key, layers)
    layers = [fnn_layer_params(width, 1, keys[0])]
    layers += [fnn_layer_params(width, width, key) for key in keys[1:-1]]
    layers += [fnn_layer_params(d, width, keys[-1])]
    return layers

def bd_loss(transform, get_latent, net_params, latent_params, bdy, t_terminal, *args):
    bdq, _ = predict_qp(transform, get_latent, net_params, latent_params, t_terminal * jnp.arange(2)[:, None], t_terminal, *args)
    return bdq - bdy

def predict_qp(transform, get_latent, net_params, latent_params, t, t_terminal, *args):
    net = sympocnet(transform, get_latent, net_params, latent_params, t, t_terminal, *args)
    return net(t)

def betal(x, l):
    return jnp.max(jnp.where(x > l, -jnp.log(jnp.clip(x, l/2)), - jnp.log(l) + 0.5 * (((x - 2*l) / l) ** 2 - 1)))

def Hamiltonian(q, p, eps, l, h):
    return jnp.sum(p ** 2) / 2 - eps * betal(h(q), l) 

@partial(vmap, in_axes = (None, 0, 0, 0, None))
def sympnet(net_params, Q, P, t, t_terminal):
    q, p = Q, P
    for i in range(len(net_params)//2):
        K, a, b = net_params[2*i]
        gradH = jnp.dot(K.T, act(jnp.dot(K, p) + b) * a)
        q = q + gradH
        K, a, b = net_params[2*i+1]
        gradH = jnp.dot(K.T, act(jnp.dot(K, q) + b) * a)
        p = p + gradH
    return q, p

def fnn(params, x, scale = 1e-2):
    y = x
    for i in range(len(params) - 1):
        W, b = params[i]
        y = act(jnp.dot(W, y) + b)
    W, b = params[-1]
    y = jnp.dot(W, y) + b
    return y * scale

def fnn_bd(params, x, left_bd = 0, right_bd = 0, scale = 1e-2):
    return fnn(params, x, scale) * x * (1 - x) + scale * (left_bd * (1 - x) + right_bd * x)

@partial(vmap, in_axes = (None, 0, 0, 0, None))
def t_sympnet(net_params, Q, P, t, t_terminal):
    q, p = Q, P
    sim_params, diag_params = net_params
    for i in range(len(diag_params)//2):
        K, b = sim_params[2*i]
        s = fnn_bd(diag_params[2*i], t/t_terminal)
        q = q + jnp.dot(K.T, (jnp.dot(K, p) + b) * s)
        K, b = sim_params[2*i+1]
        s = fnn(diag_params[2*i+1], t/t_terminal)
        p = p + jnp.dot(K.T, (jnp.dot(K, q) + b) * s)
    return q, p

def sympocnet(transform, get_latent, net_params, latent_params, t, t_terminal, *args):
    return (lambda t: transform(net_params, *get_latent(latent_params, t, t_terminal, *args), t, t_terminal))

def predict_u(transform, get_latent, params, t, t_terminal, *args):
    net_params, latent_params = params
    q, p = predict_qp(transform, get_latent, net_params, latent_params, t, t_terminal, *args)
    q, p = np.array(q), np.array(p)
    dt = t[1:] - t[:-1]
    p[:-1] = (q[1:] - q[:-1]) / dt
    p[-1] = p[-2]
    return p

def Lagrangian(v): # running cost: sum of |u|^2/2
    return jnp.sum(v**2/2, axis=-1)
  
def value_function(transform, get_latent, params, t, t_terminal, *args):
    dt = t[1:,0] - t[:-1,0]
    v = predict_u(transform, get_latent, params, t, t_terminal, *args)   
    L = Lagrangian(v)
    cost = jnp.sum(L[1:-1] * dt[1:]) + L[0]/2*dt[0] + L[-1]/2*dt[-1]
    return cost

def hmin_function(transform, get_latent, params, t, h, t_terminal, *args):
    net_params, latent_params = params
    q, _ = predict_qp(transform, get_latent, net_params, latent_params, t, t_terminal, *args)
    return jnp.min(vmap(h, in_axes = 0)(q))
