#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 19:49:46 2022

@author: zzhang99
"""
import jax.numpy as jnp
from jax import random, vmap, jit, grad, jvp, value_and_grad
from time import time
from jax.example_libraries import optimizers
from matplotlib import animation
import matplotlib.pyplot as plt
import jaxopt
import os
from net import *
from utils import mse, relu
import pickle
import argparse
import numpy as np

def init_bd():
    gridpt = jnp.linspace(-halfroomsz, halfroomsz, numdperedge + 1)
    q0 = []
    for i in range(numwall):
        drones_left = jnp.concatenate([gridpt[i]*jnp.ones([numdperedge-1-2*i,1]), gridpt[1+i:-1-i,None]], axis = -1)  # don't contain end pts
        drones_upper = jnp.concatenate([gridpt[i:numdperedge+1-i,None], gridpt[-1-i]*jnp.ones([numdperedge+1-2*i,1])], axis = -1)  # contains both end pts
        drones_left = drones_left.reshape([-1])
        drones_upper = drones_upper.reshape([-1])
        q0.extend([drones_left, drones_upper, -drones_left, -drones_upper])
    q0 = jnp.concatenate(q0, axis = 0)
    qT = -q0
    return jnp.array([q0, qT])

def dist_to_point(x):
    return jnp.sqrt(1e-10+jnp.sum((x.reshape([-1,1,phy_dim]) - ws[None,...])**2, axis = -1)) - (dr + qr)

def dist_to_segment(x):
    x = x.reshape([-1, 1, phy_dim])
    d = jnp.sum((x - vs[None,...]) * (ws - vs), axis = -1, keepdims = True) / ql ** 2 
    t = jnp.maximum(jnp.zeros_like(d), jnp.minimum(jnp.ones_like(d), d)) # num_drone * num_obs
    projection = vs[None,...] + t * (ws - vs) # ws: num_obs * 2
    d = jnp.sum((x - projection) ** 2, axis = -1)
    return jnp.sqrt(1e-10+d) - (dr + qr)

def h(x):
    x_new = x.reshape([d // phy_dim, phy_dim])
    h1 = x_new[:,0] + halfroomsz             # x >= -C
    h2 = -x_new[:,0] + halfroomsz        # x <= C
    h3 = x_new[:,1] + halfroomsz          # y >= -C
    h4 = -x_new[:,1] + halfroomsz          # y <= C
    h_bd = jnp.concatenate([h1, h2, h3, h4], axis = 0)
    h_obs = dist(x)
    h_obs = h_obs.flatten()
    # avoid hitting between drones
    x = x.reshape([d // phy_dim, phy_dim, 1])
    x = jnp.transpose(x, (1, 0, 2))
    y = jnp.transpose(x, (0, 2, 1))
    z = jnp.sum((x - y)**2, axis = 0)
    n = d // phy_dim
    min_value = z.flatten()[1:].reshape([n-1, n+1])[:,:-1]
    min_value = jnp.sqrt(min_value.flatten() + 1e-10) - (dr * 2)
    h = jnp.concatenate([h_bd, h_obs, min_value], axis = 0)
    return h

def criterion(net_params, latent_params, t):
    (q, p), (qt, pt) = my_jvp(transform, get_latent, net_params, latent_params, t, t_terminal, *args)
    dHdq, dHdp = vmap(grad(Hamiltonian, argnums=(0, 1)), in_axes = (0, 0, None, None, None))(q, p, eps, l, h)
    loss_1 = mse(qt, dHdp)
    loss_2 = mse(pt, -dHdq)
    loss_sympnet = loss_1 + loss_2
    y_m_bdq = bd_loss(transform, get_latent, net_params, latent_params, bdy, t_terminal, *args)
    loss = loss_sympnet + lam * (y_m_bdq ** 2).mean()
    #loss = loss + jnp.sum(relu(lag_mul_h - rho_h * vmap(h, in_axes = 0)(q))**2)/(2*rho_h) 
    #loss = loss + ((lag_mul_bc - rho_bc * y_m_bdq) ** 2).sum()/(2*rho_bc)
    return loss

@jit
def update(step_i, opt_state, t):
    net_params, latent_params = get_params(opt_state)
    loss, grads = value_and_grad(criterion, argnums = (0,1))(net_params, latent_params, t)
    opt_state = opt_update(step_i, grads, opt_state)
    return loss, opt_state

def train(params):
    opt_state = opt_init(params)
    t1 = time()
    min_loss = float('inf')
    best_params = None
    key = random.PRNGKey(0)
    t = jnp.sort(random.uniform(key, (num_t - 2, 1), minval = 0, maxval = t_terminal), axis = 0)
    t = jnp.concatenate([jnp.array([[0]]), t, jnp.array([[t_terminal]])], axis = 0)
    for step_i in range(steps+1):
        loss, opt_state = update(step_i, opt_state, t)
        if step_i % print_every == 0:
            t = jnp.linspace(0, t_terminal, num_t)[:, None]
            params = get_params(opt_state)
            cost = value_function(transform, get_latent, params, t, t_terminal, *args)
            hmin = hmin_function(transform, get_latent, params, t, h, t_terminal, *args)
            print('Step: ' + str(step_i))
            print('Loss: ' + str(loss))
            print('Cost value: {}\t'.format(cost))
            print('Constraint value: {}'.format(hmin))
            if loss < min_loss:
                best_params = params
                min_loss = loss
        key, _ = random.split(key)
        t = jnp.sort(random.uniform(key, (num_t - 2, 1), minval = 0, maxval = t_terminal), axis = 0)
        t = jnp.concatenate([jnp.array([[0]]), t, jnp.array([[t_terminal]])], axis = 0)
    t2 = time()
    t = jnp.linspace(0, t_terminal, num_t)[:, None]
    print('Adam time: ' + str(t2 - t1))
    criterion_lbfgs = jit(lambda params: criterion(params[0], params[1],t))
    solver = jaxopt.ScipyMinimize(fun = criterion_lbfgs, maxiter = lbfgs_steps,
                                method = 'L-BFGS-B', tol = 0, options = {'iprint':1})
    last_params, state = solver.run(best_params)
    cost = value_function(transform, get_latent, params, t, t_terminal, *args)
    hmin = hmin_function(transform, get_latent, params, t, h, t_terminal, *args)
    print('Fine tuning Loss: ' + str(state.fun_val))
    print('Cost value: {}\t'.format(cost))
    print('Constraint value: {}\n'.format(hmin))
    t3 = time()
    print('Adam time: ' + str(t2 - t1))
    print('LBFGS time: ' + str(t3 - t2))
    pickle.dump(t3 - t1, open(f'saved/time_init_{path}', "wb"))
    return last_params
            
def plot(params):
    from random import randint
    net_params, latent_params = params
    q, v = predict_qp(transform, get_latent, net_params, latent_params, t, t_terminal, *args)
    q, v = np.array(q), np.array(v)
    v0, vT = v[0], v[-1]
    pickle.dump((v0, vT), open(f'saved/init_v_{path}', "wb"))
    num_drone = d // phy_dim
    
    x = np.linspace(-halfroomsz, halfroomsz, 100)
    y = np.linspace(-halfroomsz, halfroomsz, 100)
    xx, yy = np.meshgrid(x, y)
    p = np.concatenate([xx.reshape([-1,1]), yy.reshape([-1,1])], axis = -1)
    zz = dist(p) + dr

    fig = plt.figure(figsize = [4,4])
    ax = plt.axes(xlim=(-halfroomsz*1.2, halfroomsz*1.2), ylim=(-halfroomsz*1.2, halfroomsz*1.2))
    if num_obs != 0:
        zz = np.min(zz, axis = 1).reshape([100, 100])
        plt.contour(xx,yy,zz,[0])
    colors = []
    for i in range(num_drone):
        colors.append('#%06X' % randint(0, 0xFFFFFF))
    drones = []
    for i in range(num_drone):
        drones.append(plt.Circle((q[0, 2*i], q[0, 2*i+1]), dr, fill = False, color = colors[i]))
    #for i in range(num_drone):
        #drones.append(plt.arrow(q[0, 2*i], q[0, 2*i+1], v[0, 2*i], v[0, 2*i+1], width = 0.01, fill = False, color = 'red'))
    def init():
        for i in range(num_drone):
            ax.add_patch(drones[i])
        return drones 
    def animate(i):
        for j in range(num_drone):
            drones[j].center = (q[i,2*j], q[i,2*j+1])
            #drones[num_drone+j].remove()
            #drones[num_drone+j] = plt.arrow(q[i, 2*j], q[i, 2*j+1], v[i, 2*j], v[i, 2*j+1], width = 0.01, fill = False, color = 'red')
            #ax.plot(q[:i, 2*j], q[:i, 2*j+1], c = 'gray', alpha = 0.1)
        return drones
    delay_time = 20000 // 100
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=num_t, interval=delay_time, blit=True, repeat = False)
    anim.save(f'figs/{path}.gif', writer='imagemagick', fps=30)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SympOCnet')
    parser.add_argument('--net_type', type = str, default = 'TSympNet', help = 'network type')
    parser.add_argument('--seed', default=-1, type=int, help='random seed')
    parser.add_argument('--problem', type = str, default = 'maze', help = 'free or obs or maze')
    parser.add_argument('--numdperedge', default=1, type=int, help='number of drones per edge')
    parser.add_argument('--numwall', default=1, type=int, help='number of walls')

    args = parser.parse_args()
    os.makedirs('saved', exist_ok = True)
    os.makedirs('figs', exist_ok = True)
    key = random.PRNGKey(args.seed)
    phy_dim = 2
    num_t = 200
    halfroomsz = 0.5
    numdperedge = args.numdperedge
    numwall = args.numwall
    t_terminal = 10
    t = jnp.linspace(0, t_terminal, num_t)[:, None]
    d = 8 * (numdperedge - numwall + 1) * numwall

    if args.problem == 'obs':
        dr = 0.05
        ws = jnp.array([[0,0]])
        qr = 0.3 * halfroomsz / len(ws)
        dist = dist_to_point
        eps = 1e-3
        l = 1e-3
    elif args.problem == 'free':
        if d <= 128:
            if d == 8:
                dr = 0.05
            else:
                dr = 0.02
            eps = 1e-5
            l = 1e-3
        else:
            dr = 0.01
            eps = 1e-5
            l = 1e-5
        qr = 0
        ws = jnp.empty((0, phy_dim))
        dist = dist_to_point
    elif args.problem == 'maze':
        dr = 0.02
        qr = 0.015
        ql = 2/3**0.5
        ws = np.array([[-4, -4], [-4, 0], [-4, 2], [-4 + ql/2, 3], [-4+3*ql/2, -3], [-4+3*ql/2, -1],
                  [-4+3*ql/2, 1], [-4+3*ql/2, 3], [-4+3*ql/2, -3], [-4+3*ql/2, 3], [-4+2*ql, -2],
                  [-4+2*ql, 2], [-4+3*ql, -2], [-4+3*ql, 0], [-4+3*ql, 2], [-4+7*ql/2,-3],
                  [-4+7*ql/2, 1], [-4+9*ql/2, -3], [-4+9*ql/2, -1], [-4+9*ql/2, -1],
                  [-4+9*ql/2, 1], [-4+9*ql/2, 3], [-4+5*ql, 0], [-4+6*ql, -2], [-4+6*ql, 0],
                  [-4+6*ql, 2]]) / 10
        angles = np.array([np.pi / 3, -np.pi / 3, -np.pi / 3, 0, -np.pi / 3, np.pi / 3, -np.pi / 3,
                  np.pi / 3, np.pi / 3, -np.pi / 3, 0, 0, np.pi / 3, np.pi / 3, np.pi / 3,
                  0, 0, np.pi / 3, -np.pi / 3, np.pi / 3, np.pi / 3, -np.pi / 3,
                  0, np.pi / 3, -np.pi / 3, np.pi / 3])
        ql /= 10
        vs = ws + np.stack([np.cos(angles), np.sin(angles)], axis = -1) * ql
        dist = dist_to_segment
        eps = 1e-5
        l = 1e-4

    num_obs = len(ws)
    bdy = init_bd()
    lam = 600

    path = f'{args.net_type}_{args.problem}_{args.seed}_{d}'
    os.makedirs(f'saved/{path}', exist_ok = True)
    if args.problem == 'maze':
        lr = 1e-2
        steps = 200000
    elif args.net_type in ['TSympNet', 'SympNet']:
        if d <= 128:
            lr = 1e-2
            steps = 20000
        else:
            lr = 1e-3
            steps = 100000
    else:
        lr = 1e-3
        steps = 20000

    if args.net_type == 'TSympNet':
        if args.problem != 'maze':
            width = d
            layers = 4
        else:
            width = 60
            layers = 6
        subwidth = max(100, d * 2)
        sublayers = 3
        net_params = init_tsympnet_params(d, width, subwidth, key, layers, sublayers)
        latent_params = []
        transform = t_sympnet
        get_latent = get_latent_pre
        latent_params = []
        args = [precompute_latent_fixed(bdy, jnp.linspace(0,t_terminal,1000)[:,None], t_terminal)]
        my_jvp = jvp_pre
    elif args.net_type == 'SympNet':
        width = 60
        layers = 6
        net_params = init_sympnet_params(d, width, key, layers)
        latent_params = []
        transform = sympnet
        get_latent = get_latent_pre
        latent_params = []
        args = [precompute_latent_fixed(bdy, jnp.linspace(0,t_terminal,1000)[:,None], t_terminal)]
        my_jvp = jvp_pre
    elif args.net_type == 'PINN':
        width = max(100, d * 2)
        layers = 6
        net_params = []
        latent_params = init_pinn_params(d, key, width, layers)
        transform = identity
        get_latent = pinn
        args = []
        my_jvp = jvp_pinn
    
    print_every = 1000
    train_model = True#False
    lbfgs_steps = 1000
    
    if train_model:
        params = (net_params, latent_params)
        opt_init, opt_update, get_params = optimizers.adam(lr)
        params = train(params)
        pickle.dump(params, open(f'saved/params_{path}', "wb"))
    else:
        params = pickle.load(open(f'saved/params_{path}', "rb"))

    cost = value_function(transform, get_latent, params, t, t_terminal, *args)
    print(cost)
    t1 = time()
    plot(params)
    t2 = time()
    print('Plot time: ' + str(t2 - t1))
