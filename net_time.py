from jax import random, vmap, grad, nn, jit, jvp
from functools import partial
import jax.numpy as jnp
import numpy as np
from scipy import optimize, integrate
import pickle
from math import ceil
from jax.scipy.linalg import expm, inv

def get_latent_fixed(Q_bd, t, t_terminal):
    x0, xT = Q_bd[:1, :Q_bd.shape[1] // 2], Q_bd[1:, :Q_bd.shape[1] // 2]
    a, b = -2 * (xT - x0), 3 * (xT - x0)
    new_t = t / t_terminal                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    x = a * new_t ** 3 + b * new_t ** 2 + x0
    v = (3 * a * new_t ** 2 + 2 * b * new_t) / t_terminal
    u = (6 * a * new_t + 2 * b) / t_terminal / t_terminal
    pv = u
    px = - jnp.ones_like(pv) * 6 * a / t_terminal / t_terminal / t_terminal
    return jnp.concatenate([x, v], axis = -1), jnp.concatenate([px, pv], axis = -1)

def resist(q, k):
    d = len(q) // 2
    x, v = q[:d], q[d:]
    return jnp.concatenate([jnp.zeros_like(x), -k * v * jnp.abs(v)], axis = 0)

def init_gradient_params(d, width, key, scale = 1e-2):
    K_key, a_key = random.split(key)
    return scale * random.normal(K_key, (width, d)), scale * random.normal(a_key, (width,)), jnp.zeros(width)

def init_sympnet_params(d, width, key, layers):
    keys = random.split(key, layers)
    return [init_gradient_params(d*2, width, key) for key in keys]

def init_sim_params(d, width, key, scale = 1e-2):
    return scale * random.normal(key, (width, d)), jnp.zeros(width)

def init_tsympnet_params(d, width, subwidth, key, layers, sublayers):
    sim_key, diag_key = random.split(key)
    sim_keys, diag_keys = random.split(sim_key, layers), random.split(diag_key, layers)
    return [init_sim_params(d*2, width, key) for key in sim_keys], [init_fnn_params(width, key, subwidth, sublayers) for key in diag_keys]

def fnn_layer_params(ind, outd, key):
    scale = jnp.sqrt(2 / (ind + outd))
    return random.normal(key, (ind, outd)) * scale, jnp.zeros(ind)

def init_fnn_params(d, key, width, layers):
    keys = random.split(key, layers)
    layers = [fnn_layer_params(width, 1, keys[0])]
    layers += [fnn_layer_params(width, width, key) for key in keys[1:-1]]
    layers += [fnn_layer_params(d, width, keys[-1])]
    return layers

def init_pinn_params(d, key, width, layers):
    return [init_fnn_params(d, key, width, layers), init_fnn_params(d, key, width, layers)]

def fnn(params, act, x, scale = 1e-2):
    y = x
    for i in range(len(params) - 1):
        W, b = params[i]
        y = act(jnp.dot(W, y) + b)
    W, b = params[-1]
    y = jnp.dot(W, y) + b
    return y * scale

def fnn_bd(params, act, x, left_bd = 0, right_bd = 0, scale = 1e-2):
    return fnn(params, act, x, scale) * x * (1 - x) + scale * (left_bd * (1 - x) + right_bd * x)

# PINN 
@partial(vmap, in_axes = (None, None, 0, 0, 0, None))
def identity(net_params, act, Q, P, t, t_terminal):
    return Q, P

@partial(vmap, in_axes = (None, None, 0, None))
def pinn(latent_params, act, t, t_terminal):
    new_t = t / t_terminal
    q_params, p_params = latent_params
    q = fnn(q_params, act, new_t, scale = 1)
    p = fnn(p_params, act, new_t, scale = 1)
    return q, p

def precompute_latent_fixed(d, phy_dim, Q_bd, Ai, Bi, t, h, c, path, restore = False):
    if not restore:
        q, p = get_latent_fixed(Q_bd, t, t[-1])
        qs, ps = [], []
        for i in range(d // phy_dim):
            Q_bdi = Q_bd.reshape([-1, 2, d // phy_dim, phy_dim])[:,:,i].reshape([-1, 2*phy_dim])
            H = jnp.block([[Ai, Bi @ Bi.T], [Ai.T @ Ai * c, -Ai.T]])
            H_T = expm(H * t[-1])
            p0i_new = inv(H_T[:len(H_T)//2, len(H_T)//2:]) @ (Q_bdi[1] - H_T[:len(H_T)//2, :len(H_T)//2] @ Q_bdi[0])
            q_new, p_new = dynamics_ivp(Q_bdi[0], p0i_new, Q_bdi[1], Ai, Bi, t, t[1]-t[0], phy_dim, phy_dim, 0, 1, h, 0, c, True)
            qs.append(q_new)
            ps.append(p_new)
        qs = jnp.transpose(jnp.stack(qs, axis = 1).reshape([len(t), d//phy_dim, 2, phy_dim]), (0,2,1,3)).reshape([len(t), -1])
        ps = jnp.transpose(jnp.stack(ps, axis = 1).reshape([len(t), d//phy_dim, 2, phy_dim]), (0,2,1,3)).reshape([len(t), -1])
        pickle.dump(qs, open(f'saved/qs_{path}', "wb"))
        pickle.dump(ps, open(f'saved/ps_{path}', "wb"))
        return qs, ps
    else:
        qs = pickle.load(open(f'saved/qs_{path}', "rb"))
        ps = pickle.load(open(f'saved/ps_{path}', "rb"))
        return qs, ps

def get_latent_pre(latent_params, act, t, t_terminal, qp):
    q, p = qp
    t_grid = jnp.linspace(0, t_terminal, len(q))
    q_new = vmap(lambda t, q: jnp.interp(t.flatten(), t_grid, q), in_axes = (None, 1), out_axes = 1)(t, q)
    p_new = vmap(lambda t, p: jnp.interp(t.flatten(), t_grid, p), in_axes = (None, 1), out_axes = 1)(t, p)
    #gap = ceil(len(q) / (len(t) - 1)) - 1
    #return q[::gap], p[::gap]
    return q_new, p_new

def jvp_pre(get_latent, transform, latent_params, net_params, act, t, t_terminal, A, B, c, *args):
    Q, P = get_latent(latent_params, act, t, t_terminal, *args)
    dQ = Q @ A.T + P @ B @ B.T
    dP = c * Q @ A.T @ A - P @ A
    transform_partial = (lambda Q, P, t: transform(net_params, act, Q, P, t, t_terminal))
    return jvp(transform_partial, (Q, P, t), (dQ, dP, jnp.ones_like(t)))

def jvp_pinn(get_latent, transform, latent_params, net_params, act, t, t_terminal, A, B, c, *args):
    transform_partial = lambda t: get_latent(latent_params, act, t, t_terminal, *args)
    return jvp(transform_partial, (t,), (jnp.ones_like(t),))

@partial(vmap, in_axes = (None, None, 0, 0, 0, None))
def sympnet(net_params, act, Q, P, t, t_terminal):
    q, p = Q, P
    for i in range(len(net_params)//2):
        K, a, b = net_params[2*i]
        gradH = jnp.dot(K.T, act(jnp.dot(K, p) + b) * a)
        q = q + gradH
        K, a, b = net_params[2*i+1]
        gradH = jnp.dot(K.T, act(jnp.dot(K, q) + b) * a)
        p = p + gradH
    return q, p

@partial(vmap, in_axes = (None, None, 0, 0, 0, None))
def t_sympnet(net_params, act, Q, P, t, t_terminal):
    q, p = Q, P
    sim_params, diag_params = net_params
    for i in range(len(diag_params)//2):
        K, b = sim_params[2*i]
        s = fnn_bd(diag_params[2*i], act, t/t_terminal)
        q = q + jnp.dot(K.T, (jnp.dot(K, p) + b) * s)
        K, b = sim_params[2*i+1]
        s = fnn(diag_params[2*i+1], act, t/t_terminal)
        p = p + jnp.dot(K.T, (jnp.dot(K, q) + b) * s)
    return q, p

def sympocnet(act, transform, get_latent, net_params, latent_params, t, t_terminal, *args):
    return (lambda t: transform(net_params, act, *get_latent(latent_params, act, t, t_terminal, *args), t, t_terminal))

def predict_qp(act, transform, get_latent, net_params, latent_params, t, t_terminal, *args):
    net = sympocnet(act, transform, get_latent, net_params, latent_params, t, t_terminal, *args)
    return net(t)

def predict_qp_from_sympnet(act, transform, get_latent, net_params, latent_params, t, t_terminal, k, *args):
    dt = t[1:] - t[:-1]
    net = sympocnet(act, transform, get_latent, net_params, latent_params, t, t_terminal, *args)
    q, p = net(t)
    q, p = np.array(q), np.array(p)
    d = q.shape[1]
    q[:-1, d//2:] = (q[1:, :d//2] - q[:-1, :d//2]) / dt
    q[-1, d//2:] = q[-2, d//2:]
    p[:, :d//2] = 0
    p[:-1, d//2:] = ((q[1:, d//2:] - q[:-1, d//2:]) / dt - resist(q, k)[:-1, d//2:]) 
    p[-1, d//2:] = p[-2, d//2:]
    return q, p 

def predict_qp_from_dynamics(act, transform, get_latent, net_params, latent_params, t, t_terminal, A, B, k, *args):
    net = sympocnet(act, transform, get_latent, net_params, latent_params, t, t_terminal, *args)
    q, p = net(t)
    u = predict_u(act, transform, get_latent, (net_params, latent_params), t, B, t_terminal, *args)
    dt = t[1] - t[0]
    q = [q[0]]
    for i in range(len(t) - 1):
        q.append(q[-1] + dt * (A @ q[-1] + resist(q[-1], k) + B @ u[i]))
    q = jnp.stack(q, axis = 0)
    return q, p

def predict_qp_from_Hamiltonian(act, transform, get_latent, net_params, latent_params, t, A, B, phy_dim, d, eps, l, h, Q_bd, k, c, tune_with_shooting, *args):
    t_terminal = t[-1, 0]
    q, p = predict_qp(act, transform, get_latent, net_params, latent_params, t, t_terminal, *args)
    q0, p0 = Q_bd[0], p[0]
    qT = Q_bd[1]
    if k == 2:
        outer_step = 1
    elif c > 0:
        outer_step = 2
    else:
        outer_step = 10
    decay_rate = 0.8
    if tune_with_shooting:
        p0 = shooting(Q_bd, p[0], A, B, t, phy_dim, d, eps, l, h, k, c, outer_step, decay_rate)
    dt = t[1] - t[0]
    q, p = dynamics_ivp(q0, p0, qT, A, B, t, dt, phy_dim, d, eps*decay_rate**(outer_step-1), l*decay_rate**(outer_step-1), h, k, c, True)
    print(q)
    print(q.shape)
    return q, p

def betal(x, l):
    return jnp.max(jnp.where(x > l, -jnp.log(jnp.clip(x, l/2)), - jnp.log(l) + 0.5 * (((x - 2*l) / l) ** 2 - 1)), initial = 0)

def Hamiltonian(q, p, phy_dim, d, eps, l, h, A, B, k, c):
    H1 = jnp.sum(p * (A @ q + resist(q, k)))
    H2 = jnp.sum((B.T @ p)**2) / 2
    H3 = - eps * betal(h(q), l) 
    H4 = - c * jnp.sum((A @ q) ** 2) / 2
    return H1 + H2 + H3 + H4

def bd_loss(act, transform, get_latent, net_params, latent_params, Q_bd, t_terminal, *args):
    Q_bd_pred, _ = predict_qp(act, transform, get_latent, net_params, latent_params, jnp.arange(2)[:, None] * t_terminal, t_terminal, *args)
    return Q_bd_pred - Q_bd

def predict_u(act, transform, get_latent, params, t, B, t_terminal, k, *args):
    net_params, latent_params = params
    _, p = predict_qp_from_sympnet(act, transform, get_latent, net_params, latent_params, t, t_terminal, k, *args)
    u = p @ B
    return u

def predict_v(act, transform, get_latent, params, t, B, t_terminal, k, *args):
    net_params, latent_params = params
    q, _ = predict_qp_from_sympnet(act, transform, get_latent, net_params, latent_params, t, t_terminal, k, *args)
    return q[:, q.shape[1]//2:]

def Lagrangian(u, v, c): # running cost: sum of |u|^2/2
    return jnp.sum(u**2/2, axis=-1) + jnp.sum(v**2/2*c, axis=-1)
  
def value_function(act, transform, get_latent, params, t, B, t_terminal, c, k, *args):
    dt = t[1:,0] - t[:-1,0]
    u = predict_u(act, transform, get_latent, params, t, B, t_terminal, k, *args)   
    v = predict_v(act, transform, get_latent, params, t, B, t_terminal, k, *args) 
    L = Lagrangian(u, v, c)
    cost = jnp.sum(L[1:-1] * dt[1:]) + L[0]/2*dt[0] + L[-1]/2*dt[-1]
    return cost

def hmin_function(act, transform, get_latent, params, t, h, t_terminal, *args):
    net_params, latent_params = params
    q, _ = predict_qp(act, transform, get_latent, net_params, latent_params, t, t_terminal, *args)
    return jnp.min(vmap(h, in_axes = 0)(q))

def dynamics_ivp(q0, p0, qT, A, B, t, dt, phy_dim, d, eps, l, h, k, c, return_all = False):
    @jit
    def f(t, x):
        q, p = x[:len(x)//2], x[len(x)//2:]
        dH = grad(Hamiltonian, argnums=(0, 1))(q, p, phy_dim, d, eps, l, h, A, B, k, c)
        return jnp.concatenate([dH[1], -dH[0]])
    if return_all:
        qp = integrate.solve_ivp(f, [t[0], t[-1]], jnp.concatenate([q0,p0]), t_eval = t.flatten()).y.T
        return qp[:, :len(q0)], qp[:, len(q0):]
    else:
        q = integrate.solve_ivp(f, [t[0], t[-1]], jnp.concatenate([q0,p0])).y[:len(q0),-1]
        return q - qT

def shooting(Q_bd, p0, A, B, t, phy_dim, d, eps, l, h, k, c, outer_step, decay_rate):
    q0, qT = Q_bd
    dt = t[1] - t[0]
    factor = 0.01
    for i in range(outer_step):
        objective = lambda p0: dynamics_ivp(q0, p0, qT, A, B, t, dt, phy_dim, d, eps, l, h, k, c)
        res = optimize.fsolve(objective, p0, factor = factor)#np.random.randn(*p0.shape))
        print(objective(res))
        print(res - p0)
        p0 = np.array(res)
        l *= decay_rate
        eps *= decay_rate
    return p0


