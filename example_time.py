import jax.numpy as jnp
from jax import random, vmap, jit, grad, jvp, value_and_grad
from time import time
from jax.example_libraries import optimizers
from matplotlib import animation
import matplotlib.pyplot as plt
import jaxopt
import os
from net_time import *
from utils import mse, relu
import pickle
import argparse
import numpy as np

def init_bd_simple(v0, vT):
    q0 = jnp.array([-0.5, 0.5])
    qT = -q0
    if v0 is None:
        v0, vT = jnp.zeros_like(q0), jnp.zeros_like(qT)
    q0 = jnp.concatenate([q0, v0])
    qT = jnp.concatenate([qT, vT])
    return jnp.array([q0, qT])

def init_bd(v0, vT):
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
    if v0 is None:
        v0, vT = jnp.zeros_like(q0), jnp.zeros_like(qT)
    q0 = jnp.concatenate([q0, v0])
    qT = jnp.concatenate([qT, vT])
    return jnp.array([q0, qT])

def update_v(Q_bd):
    q_bd = Q_bd[:, :Q_bd.shape[1] // 2]
    v_bd = Q_bd[:, Q_bd.shape[1] // 2:]
    if v0 is not None:
        v_bd = v_bd - jnp.array([v0, vT]) / (outer_steps - 1)
    return jnp.concatenate([q_bd, v_bd], axis = 1)

def dist_to_point(x):
    return jnp.sqrt(1e-10+jnp.sum((x.reshape([-1,1,phy_dim]) - ws[None,...])**2, axis = -1)) - (dr + qr)

def dist_to_segment(x):
    x = x.reshape([-1, 1, phy_dim])
    d = jnp.sum((x - vs[None,...]) * (ws - vs), axis = -1, keepdims = True) / ql ** 2 
    t = jnp.maximum(jnp.zeros_like(d), jnp.minimum(jnp.ones_like(d), d)) # num_drone * num_obs
    projection = vs[None,...] + t * (ws - vs) # ws: num_obs * 2
    d = jnp.sum((x - projection) ** 2, axis = -1)
    return jnp.sqrt(1e-10+d) - (dr + qr)

def h(q):
    d = len(q) // 2
    x, v = q[:d], q[d:]
    x_new = x.reshape([d // phy_dim, phy_dim])
    h1 = x_new[:,0] + halfroomsz * 1.05          # x >= -C
    h2 = -x_new[:,0] + halfroomsz * 1.05       # x <= C
    h3 = x_new[:,1] + halfroomsz * 1.05         # y >= -C
    h4 = -x_new[:,1] + halfroomsz * 1.05         # y <= C
    h_bd = jnp.concatenate([h1, h2, h3, h4], axis = 0)
    h_obs = dist(x)
    h_obs = h_obs.flatten()
    hv = Cv - jnp.sqrt(1e-10+jnp.sum(v.reshape([d // phy_dim, phy_dim]) ** 2, axis = -1))
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
    Q, P = get_latent(latent_params, act, t, t_terminal, *args)
    (q, p), (qt, pt) = my_jvp(get_latent, transform, latent_params, net_params, act, t, t_terminal, A, B, c, *args)
    #sympnet_partial = sympocnet(act, transform, get_latent, net_params, latent_params, t, t_terminal, Q_bd, m, *args)
    #(q, p), (qt, pt) = jvp(sympnet_partial, (t,), (jnp.ones_like(t),))
    dHdq, dHdp = vmap(grad(Hamiltonian, argnums=(0, 1)), in_axes = (0, 0, None, None, None, None, None, None, None, None, None))(q, p, phy_dim, d, eps, l, h, A, B, k, c)
    loss_1 = mse(qt, dHdp)
    loss_2 = mse(pt, -dHdq)
    #v_fd = (q[1:, :d] - q[:-1, :d]) / dt
    #hv = Cv - jnp.sqrt(1e-10+jnp.sum(v_fd.reshape([-1, d // phy_dim, phy_dim]) ** 2, axis = -1))
    #loss_penalty = eps * betal(hv, l) 
    loss = loss_1 + loss_2
    loss_bd = lam * (bd_loss(act, transform, get_latent, net_params, latent_params, Q_bd, t_terminal, *args) ** 2).mean()
    return loss + loss_bd

@jit
def update(step_i, opt_state, t):
    net_params, latent_params = get_params(opt_state)
    loss, grads = value_and_grad(lambda x, y: criterion(x, y, t), argnums = (0, 1))(net_params, latent_params)
    opt_state = opt_update(step_i, grads, opt_state)
    return loss, opt_state

def train(outer_step, params):
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
            cost = value_function(act, transform, get_latent, params, t, B, t_terminal, c, k, *args)
            hmin = hmin_function(act, transform, get_latent, params, t, h, t_terminal, *args)
            loss = criterion(params[0], params[1], t)
            print('Step: ' + str(step_i))
            print('Loss: ' + str(loss))
            print('Cost value: {}\t'.format(cost))
            print('Constraint value: {}'.format(hmin))
            print()
            if loss < min_loss:
                best_params = params
                min_loss = loss
        key, _ = random.split(key)
        t = jnp.sort(random.uniform(key, (num_t - 2, 1), minval = 0, maxval = t_terminal), axis = 0)
        t = jnp.concatenate([jnp.array([[0]]), t, jnp.array([[t_terminal]])], axis = 0)
    t2 = time()
    t = jnp.linspace(0, t_terminal, num_t)[:, None]
    criterion_lbfgs = jit(lambda params: criterion(params[0], params[1], t))
    solver = jaxopt.ScipyMinimize(fun = criterion_lbfgs, maxiter = lbfgs_steps,
                                method = 'L-BFGS-B', tol = 0, options = {'iprint':1})
    params, state = solver.run(best_params)
    cost = value_function(act, transform, get_latent, params, t, B, t_terminal, c, k, *args)
    hmin = hmin_function(act, transform, get_latent, params, t, h, t_terminal, *args)
    print('Fine tuning Loss: ' + str(state.fun_val))
    print('Cost value: {}\t'.format(cost))
    print('Constraint value: {}\n'.format(hmin))
    t3 = time()
    print('Adam time: ' + str(t2 - t1)) 
    print('LBFGS time: ' + str(t3 - t2))
    return params, cost, hmin, state.fun_val
            
def plot(params):
    net_params, latent_params = params
    q, p = predict_qp(act, transform, get_latent, net_params, latent_params, t, t_terminal, *args)
    #q, p = predict_qp_from_sympnet(act, transform, get_latent, net_params, latent_params, t, t_terminal, k, *args)
    #q, p = predict_qp_from_dynamics(act, transform, get_latent, net_params, latent_params, t, t_terminal, A, B, k, *args)
    #q, p = predict_qp_from_Hamiltonian(act, transform, get_latent, net_params, latent_params, t, A, B, phy_dim, d, eps, l, h, k, c, tune_with_shooting, *args)
    pickle.dump((q, p), open(f'saved/predicted_qp_{path}_old', "wb"))
    q, v = q[:, :d], q[:, d:]
    u = p[:, d:]
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
    drones = []
    for i in range(num_drone):
        drones.append(plt.Circle((q[0, 2*i], q[0, 2*i+1]), dr, fill = False, color = 'black'))
    for i in range(num_drone):
        drones.append(plt.arrow(q[0, 2*i], q[0, 2*i+1], v[0, 2*i], v[0, 2*i+1], width = 0.01, fill = False, color = 'red'))
    for i in range(num_drone):
        drones.append(plt.arrow(q[0, 2*i], q[0, 2*i+1], u[0, 2*i], u[0, 2*i+1], width = 0.01, fill = False, color = 'blue'))
    def init():
        for i in range(num_drone):
            ax.add_patch(drones[i])
        return drones 
    def animate(i):
        for j in range(num_drone):
            drones[j].center = (q[i,2*j], q[i,2*j+1])
            drones[num_drone+j].remove()
            drones[num_drone*2+j].remove()
            drones[num_drone+j] = plt.arrow(q[i, 2*j], q[i, 2*j+1], v[i, 2*j], v[i, 2*j+1], width = 0.01, fill = False, color = 'red')
            drones[num_drone*2+j] = plt.arrow(q[i, 2*j], q[i, 2*j+1], u[i, 2*j], u[i, 2*j+1], width = 0.01, fill = False, color = 'blue')
            ax.plot(q[:i, 2*j], q[:i, 2*j+1], c = 'gray', alpha = 0.1)
        return drones
    delay_time = 20000 // num_t
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(q), interval=delay_time, blit=True, repeat = False)
    anim.save(f'figs/{path}.gif', writer='imagemagick', fps=30)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SympOCnet')
    parser.add_argument('--seed', default=-1, type=int, help='random seed')
    parser.add_argument('--numdperedge', default=1, type=int, help='number of drones per edge')
    parser.add_argument('--numwall', default=1, type=int, help='number of walls')
    parser.add_argument('--Cv', default=0.5, type=float, help='speed limit')
    parser.add_argument('--k', default=0.0, type=float, help='resistent coefficient')
    parser.add_argument('--c', default=0.0, type=float, help='v penalty coefficient')
    parser.add_argument('--problem', type = str, default = 'maze', help = 'free or obs or maze')
    parser.add_argument('--net_type', type = str, default = 'TSympNet', help = 'network type')
    parser.add_argument('--latent', type = str, default = 'pre', help = 'fixed or free')
    parser.add_argument('--act', type = str, default = 'swish', help = 'tanh or relu or sigmoid')
    args = parser.parse_args()

    os.makedirs('saved', exist_ok = True)
    os.makedirs('figs', exist_ok = True)
    seed = args.seed
    numdperedge = args.numdperedge
    numwall = args.numwall
    problem = args.problem
    Cv = args.Cv
    k = args.k
    c = args.c
    act_fun = args.act
    net_type = args.net_type
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
    latent = args.latent
    key = random.PRNGKey(seed)

    phy_dim = 2
    num_t = 200
    halfroomsz = 0.5
    t_terminal = 10
    dt = t_terminal / (num_t - 1)
    if numdperedge == 0:
        d = 2
    else:
        d = 8 * (numdperedge - numwall + 1) * numwall

    #ws = jnp.empty((0, phy_dim))
    if problem == 'obs':
        eps = 1e-3
        l = 1e-3
        dr = 0.05
        ws = jnp.array([[0,0]])
        qr = 0.3 * halfroomsz / len(ws)
        dist = dist_to_point
    elif problem == 'free':
        if d <= 128:
            if d == 8:
                dr = 0.1
            else:
                dr = 0.02
            eps = 1e-5
        else:
            dr = 0.01
            eps = 5e-4
        l = 1e-3
        qr = 0
        ws = jnp.empty((0, phy_dim))
        dist = dist_to_point
    elif problem == 'maze':
        eps = 1e-5
        l = 1e-4
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

    num_obs = len(ws)
    t = jnp.linspace(0, t_terminal, num_t)[:, None]
    train_model = True#False
    if problem == 'obs':
        tune_with_shooting = True
    else:
        tune_with_shooting = False

    short_path = f'{args.net_type}_{args.problem}_{args.seed}_{d}'
    if train_model and os.path.exists(f'saved/init_v_{short_path}') and net_type == 'free':
        v0, vT = pickle.load(open(f'saved/init_v_{short_path}', "rb"))
        v0 /= t_terminal
        vT /= t_terminal
    else:
        v0, vT = None, None
    if numdperedge == 0:
        Q_bd = init_bd_simple(v0, vT)
    else:
        Q_bd = init_bd(v0, vT)

    A = jnp.concatenate([jnp.zeros([d, d]), jnp.eye(d, d)], axis = 1)
    A = jnp.concatenate([A, jnp.zeros_like(A)], axis = 0)
    B = jnp.concatenate([jnp.zeros([d, d]), jnp.eye(d, d)], axis = 0)
    Ai = jnp.concatenate([jnp.zeros([phy_dim, phy_dim]), jnp.eye(phy_dim, phy_dim)], axis = 1)
    Ai = jnp.concatenate([Ai, jnp.zeros_like(Ai)], axis = 0)
    Bi = jnp.concatenate([jnp.zeros([phy_dim, phy_dim]), jnp.eye(phy_dim, phy_dim)], axis = 0)
    path = f'{problem}_{net_type}_{seed}_{d}_{Cv}_{k}_{c}'
    print(path)
    
    if net_type == 'TSympNet':
        if args.problem != 'maze':
            width = d * 2
            layers = 4
        else:
            width = 60
            layers = 6
        subwidth = max(100, d * 2)
        sublayers = 3
        net_params = init_tsympnet_params(d, width, subwidth, key, layers, sublayers)
        transform = t_sympnet
        my_jvp = jvp_pre
        get_latent = get_latent_pre
        latent_params = []
    elif net_type == 'SympNet':
        width = 60
        layers = 6
        net_params = init_sympnet_params(d, width, key, layers)
        transform = sympnet
        my_jvp = jvp_pre
        get_latent = get_latent_pre
        latent_params = []
    elif net_type == 'PINN':
        width = max(100, d * 2)
        layers = 6
        net_params = []
        latent_params = init_pinn_params(d * 2, key, width, layers)
        transform = identity
        get_latent = pinn
        my_jvp = jvp_pinn

    lam = 600
    print_every = 1000
    if net_type in ['TSympNet', 'SympNet'] and d <= 128 and problem == 'free':
        lr = 1e-2
    else:
        lr = 1e-3
    lbfgs_steps = 1000
    if problem == 'obs':
        steps = 20000
        outer_steps = 5
        decay_rate = 0.8
    elif problem == 'free':
        steps = 20000
        outer_steps = 5
        decay_rate = 1
    else:
        steps = 300000
        outer_steps = 50
        decay_rate = 1
    os.makedirs(f'saved/{path}', exist_ok = True)
    t0 = time()
    if problem in ['maze', 'free']:
        best_criterion = 'cost'
    else:
        best_criterion = 'loss'

    if train_model:
        params = (net_params, latent_params)
        opt_init, opt_update, get_params = optimizers.adam(lr)
        min_criterion = float('inf')
        best_params = params
        for i in range(outer_steps):
            if net_type in ['TSympNet', 'SympNet']:
                args = [precompute_latent_fixed(d, phy_dim, Q_bd, Ai, Bi, jnp.linspace(0,t_terminal,1000)[:,None], h, c, path, restore = False)]
            else:
                args = []
            params, cost, hmin, loss = train(i, params)
            old_eps, old_l = eps, l
            eps, l = 1e-4, 1e-4
            true_loss = criterion(*params, t)
            eps, l = old_eps, old_l

            if i < outer_steps - 1:
                eps *= decay_rate
                l *= decay_rate
                Q_bd = update_v(Q_bd)
            if (true_loss < min_criterion and best_criterion == 'loss' and hmin > 8e-3):
                min_criterion = loss
                best_params = params
            if (cost < min_criterion and best_criterion == 'cost' and hmin > 8e-3):
                min_criterion = cost
                best_params = params
        pickle.dump(best_params, open(f'saved/best_params_{path}', "wb"))
        pickle.dump(params, open(f'saved/last_params_{path}', "wb"))
        #params = best_params
    else:
        if net_type in ['TSympNet', 'SympNet']:
            args = [precompute_latent_fixed(d, phy_dim, Q_bd, Ai, Bi, jnp.linspace(0,t_terminal,1000)[:,None], h, c, path, restore = False)]
        else:
            args = []
        eps *= decay_rate ** (outer_steps - 1)
        l *= decay_rate ** (outer_steps - 1)
        params = pickle.load(open(f'saved/last_params_{path}', "rb"))
    
    cost = value_function(act, transform, get_latent, params, t, B, t_terminal, c, k, *args)
    print(f'Cost: {cost}')
    hmin = hmin_function(act, transform, get_latent, params, t, h, t_terminal, *args)
    print(f'Constraint: {hmin}')
    loss = criterion(*params, t)
    print(f'Loss: {loss}')

    t1 = time()
    print('Training time: ' + str(t1 - t0))
    #init_time = pickle.load(open(f'saved/time_init_{short_path}', "rb"))
    #total_time = t1 - t0 + init_time
    #if train_model:
        #pickle.dump(total_time, open(f'saved/time_{path}', "wb"))
        #np.savetxt(f'saved/cost_{path}.txt', np.array([cost]))
        #np.savetxt(f'saved/hmin_{path}.txt', np.array([hmin]))
    #cost = np.loadtxt(f'saved/cost_{path}.txt')
    #print(cost)
    plot(params)
    t2 = time()
    print('Plot time: ' + str(t2 - t1))
