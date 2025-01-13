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

def init_bd(v0, vT):
    qT = np.array([-2.5, 2.5, 6.5,
                           -2., 2., 6.,
                           -1.5, 2.5, 6.5, 
                               -1., 2., 6.,
                               -0.5, 2.5, 6.5,
                               0., 2., 6.,
                               0.5, 2.5, 6.5,
                               1., 2., 6.,
                               1.5, 2.5, 6.5,
                               2., 2., 6.,
                               2.5, 2.5, 6.5,
                               3., 2., 6.,
                               3.5, 2.5, 6.5,
                               4., 2., 6.,
                               4.5, 2.5, 6.5,
                               -2.5, 3., 7.,
                               -2, 3.5, 7.5,
                               -1.5, 3., 7.,
                               -1, 3.5, 7.5,
                               -0.5, 3., 7.,
                               0., 3.5, 7.5,
                               0.5, 3., 7.,
                               1., 3.5, 7.5,
                               1.5, 3., 7.,
                               2., 3.5, 7.5,
                               2.5, 3., 7.,
                               3., 3.5, 7.5,
                               3.5, 3., 7.,
                               -2.5, 4.5, 8.5,
                               -2., 4., 8.,
                               -1.5, 4.5, 8.5,
                               -1., 4., 8.,
                               -0.5, 4.5, 8.5,
                               0., 4., 8.,
                               0.5, 4.5, 8.5,
                               1., 4., 8.,
                               1.5, 4.5, 8.5,
                               2., 4., 8.,
                               2.5, 4.5, 8.5,
                               3., 4., 8.,
                               3.5, 4.5, 8.5,
                               4., 4., 8.,
                               -2.5, 3.5, 5.5,
                               -2., 3., 5.,
                               -1.5, 3.5, 5.5,
                               -1., 3., 5.,
                               0, 3.5, 5.5,
                               1., 3., 5.,
                               1.5, 3.5, 5.5,
                               2., 3., 5.]).reshape([-1,3])
    qT = np.concatenate((qT, np.array([0, -0.5, -3]) + qT), axis=0)
    q0 = np.array([1,-1,-1]) * qT + np.array([0,0,10])
    q0, qT = q0.flatten(), qT.flatten()
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

def h(q):
    def compute_single_barrier(barrier):
        h1 = jnp.maximum(x_new[:, 0] - barrier[0], -x_new[:, 0] + barrier[1])
        h2 = jnp.maximum(x_new[:, 1] - barrier[2], -x_new[:, 1] + barrier[3])
        h3 = jnp.maximum(x_new[:, 2] - barrier[4], -x_new[:, 2] + barrier[5])
        return jnp.max(jnp.stack([h1, h2, h3]), axis=0)
    d = len(q) // 2
    x, v = q[:d], q[d:]
    x_new = x.reshape([d // phy_dim, phy_dim])
    h = vmap(compute_single_barrier)(barriers)
    h = jnp.concatenate(h, axis=0)
    # avoid hitting between drones
    x = x.reshape([d // phy_dim, phy_dim, 1])
    x = jnp.transpose(x, (1, 0, 2))
    y = jnp.transpose(x, (0, 2, 1))
    z = jnp.sum((x - y)**2, axis = 0)
    n = d // phy_dim
    min_value = z.flatten()[1:].reshape([n-1, n+1])[:,:-1]
    min_value = jnp.sqrt(min_value.flatten() + 1e-10) - (dr * 2)
    h = jnp.concatenate([h, min_value], axis = 0)
    return h

def criterion(net_params, latent_params, t):
    Q, P = get_latent(latent_params, act, t, t_terminal, *args)
    (q, p), (qt, pt) = my_jvp(get_latent, transform, latent_params, net_params, act, t, t_terminal, A, B, c, *args)
    dHdq, dHdp = vmap(grad(Hamiltonian, argnums=(0, 1)), in_axes = (0, 0, None, None, None, None, None, None, None, None, None))(q, p, phy_dim, d, eps, l, h, A, B, k, c)
    loss_1 = mse(qt, dHdp)
    loss_2 = mse(pt, -dHdq)
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

def plot_swarm(params):
    from random import randint
    from mpl_toolkits.mplot3d import Axes3D
    net_params, latent_params = params
    q, p = predict_qp(act, transform, get_latent, net_params, latent_params, t, t_terminal, *args)
    q, v = q[:, :d], q[:, d:]
    q, v = np.array(q), np.array(v)
    num_drone = d // phy_dim
    qT = Q_bd[1].reshape([-1,phy_dim])
    t_grid = [int(tt * (num_t - 1)) for tt in [1/3,2/3,1]]
    colors = []
    for i in range(num_drone):
        colors.append('#%06X' % randint(0, 0xFFFFFF))
    #num_drone = net.dim // 2
    for j in range(3):
        fig = plt.figure(figsize = [5,5])
        ax = plt.axes(projection = '3d')
        ax.set_xlim3d([-4.0, 5.0])
        ax.set_xlabel('X')
        ax.set_ylim3d([-5.0, 5.0])
        ax.set_ylabel('Y')
        ax.set_zlim3d([0.0, 9.0])
        ax.set_zlabel('Z')
        shade = 0.4
        # block 1
        X, Y = np.meshgrid([-1.8, 1.8], [-0.3, 0.3])
        ax.plot_surface(X, Y, 7 * np.ones((2,2)) , alpha=shade, color='gray')
        ax.plot_surface(X, Y, 0 * np.ones((2, 2)), alpha=shade, color='gray')
        X, Z = np.meshgrid([-1.8, 1.8], [0.2, 6.8])
        ax.plot_surface(X, -0.5 * np.ones((2, 2)), Z, alpha=shade, color='gray')
        ax.plot_surface(X,  0.5 * np.ones((2, 2)), Z, alpha=shade, color='gray')
        Y, Z = np.meshgrid([-0.3, 0.3], [0.2, 6.8])
        ax.plot_surface(-2 * np.ones((2, 2)), Y , Z, alpha=shade, color='gray')
        ax.plot_surface( 2 * np.ones((2, 2)), Y , Z, alpha=shade, color='gray')
        # block 2
        X, Y = np.meshgrid([2.2, 3.8], [-0.8, 0.8])
        ax.plot_surface(X, Y, 4 * np.ones((2,2)) , alpha=shade, color='gray')
        ax.plot_surface(X, Y, 0 * np.ones((2, 2)), alpha=shade, color='gray')
        X, Z = np.meshgrid([2.2, 3.8], [0.2, 3.8])
        ax.plot_surface(X, -1 * np.ones((2, 2)), Z, alpha=shade, color='gray')
        ax.plot_surface(X,  1 * np.ones((2, 2)), Z, alpha=shade, color='gray')
        Y, Z = np.meshgrid([-0.8, 0.8], [0.2, 3.8])
        ax.plot_surface( 2 * np.ones((2, 2)), Y , Z, alpha=shade, color='gray')
        ax.plot_surface( 4 * np.ones((2, 2)), Y , Z, alpha=shade, color='gray')
        ax.plot(qT[:,0], qT[:,1], qT[:,2], 'rx')
        ax.view_init(60, -20)
        for i in range(num_drone):
            ax.plot(q[:t_grid[j], 3*i], q[:t_grid[j], 3*i+1], q[:t_grid[j], 3*i+2], c = colors[i])
        plt.tight_layout()
        plt.savefig(f'figs/3d_{path}_{j}.pdf')
        plt.show()
        plt.close()
            
def plot(params):
    from random import randint
    from mpl_toolkits.mplot3d import Axes3D
    net_params, latent_params = params
    q, p = predict_qp(act, transform, get_latent, net_params, latent_params, t, t_terminal, *args)
    q, v = q[:, :d], q[:, d:]
    q, v = np.array(q), np.array(v)
    num_drone = d // phy_dim
    qT = Q_bd[1].reshape([-1,phy_dim])
    fig = plt.figure(figsize = [8.5,5])
    ax = plt.axes(projection = '3d')
    ax.set_xlim3d([-4.0, 5.0])
    ax.set_xlabel('X')
    ax.set_ylim3d([-5.0, 5.0])
    ax.set_ylabel('Y')
    ax.set_zlim3d([0.0, 9.0])
    ax.set_zlabel('Z')
    shade = 0.4
    # block 1
    X, Y = np.meshgrid([-2, 2], [-0.5, 0.5])
    ax.plot_surface(X, Y, 7 * np.ones((2,2)) , alpha=shade, color='gray')
    ax.plot_surface(X, Y, 0 * np.ones((2, 2)), alpha=shade, color='gray')
    X, Z = np.meshgrid([-2, 2], [0, 7])
    ax.plot_surface(X, -0.5 * np.ones((2, 2)), Z, alpha=shade, color='gray')
    ax.plot_surface(X,  0.5 * np.ones((2, 2)), Z, alpha=shade, color='gray')
    Y, Z = np.meshgrid([-0.5, 0.5], [0, 7])
    ax.plot_surface(-2 * np.ones((2, 2)), Y , Z, alpha=shade, color='gray')
    ax.plot_surface( 2 * np.ones((2, 2)), Y , Z, alpha=shade, color='gray')
    # block 2
    X, Y = np.meshgrid([2, 4], [-1, 1])
    ax.plot_surface(X, Y, 4 * np.ones((2,2)) , alpha=shade, color='gray')
    ax.plot_surface(X, Y, 0 * np.ones((2, 2)), alpha=shade, color='gray')
    X, Z = np.meshgrid([2, 4], [0, 4])
    ax.plot_surface(X, -1 * np.ones((2, 2)), Z, alpha=shade, color='gray')
    ax.plot_surface(X,  1 * np.ones((2, 2)), Z, alpha=shade, color='gray')
    Y, Z = np.meshgrid([-1, 1], [0, 4])
    ax.plot_surface( 2 * np.ones((2, 2)), Y , Z, alpha=shade, color='gray')
    ax.plot_surface( 4 * np.ones((2, 2)), Y , Z, alpha=shade, color='gray')
    ax.plot(qT[:,0], qT[:,1], qT[:,2], 'rx')
    ax.view_init(60, -30)
    colors = []
    for i in range(num_drone):
        colors.append('#%06X' % randint(0, 0xFFFFFF))
    drones = []
    for i in range(num_drone):
        drone, = ax.plot(q[:1, 3*i], q[:1, 3*i+1], q[:1, 3*i+2], c = colors[i])
        drones.append(drone)
    def init():
        for drone in drones:
            drone.set_data([],[])
        return drones
    def animate(i):
        for j in range(num_drone):
            drones[j].set_data(q[:i, 3*j], 
                    q[:i, 3*j+1])
            drones[j].set_3d_properties(q[:i, 3*j+2])
        return drones
    frames = q.shape[0]
    delay_time = 20000 // q.shape[0]
    anim = animation.FuncAnimation(fig, animate, init_func = init,
                               frames=frames, interval=delay_time, blit=True, repeat = False)
    anim.save(f'figs/3d_{path}.gif', writer='imagemagick', fps=30)
    plt.show()
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SympOCnet')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
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
    barriers = jnp.array([[2, -2, 0.5, -0.5, 7, 0], [4, 2, 1, -1, 4, 0]])

    train_model = False
    tune_with_shooting = False#True
    phy_dim = 3
    num_t = 200
    t_terminal = 10
    dt = t_terminal / (num_t - 1)
    short_path = f'{args.net_type}_{args.seed}_300'
    if train_model and os.path.exists(f'saved/init_v_{short_path}'):
        v0, vT = pickle.load(open(f'saved/init_v_{short_path}', "rb"))
    else:
        v0, vT = None, None
    Q_bd = init_bd(v0, vT)
    d = Q_bd.shape[1] // 2
    eps = 1e-5
    l = 1e-4
    dr = 0.2

    t = jnp.linspace(0, t_terminal, num_t)[:, None]
    A = jnp.concatenate([jnp.zeros([d, d]), jnp.eye(d, d)], axis = 1)
    A = jnp.concatenate([A, jnp.zeros_like(A)], axis = 0)
    B = jnp.concatenate([jnp.zeros([d, d]), jnp.eye(d, d)], axis = 0)
    Ai = jnp.concatenate([jnp.zeros([phy_dim, phy_dim]), jnp.eye(phy_dim, phy_dim)], axis = 1)
    Ai = jnp.concatenate([Ai, jnp.zeros_like(Ai)], axis = 0)
    Bi = jnp.concatenate([jnp.zeros([phy_dim, phy_dim]), jnp.eye(phy_dim, phy_dim)], axis = 0)
    path = f'{problem}_{net_type}_{seed}_{d}_{Cv}_{k}_{c}'
    print(path)
    width = 60
    layers = 6
    subwidth = max(100, d * 2)
    sublayers = 3
    net_params = init_tsympnet_params(d, width, subwidth, key, layers, sublayers)
    transform = t_sympnet
    my_jvp = jvp_pre
    get_latent = get_latent_pre
    latent_params = []

    lam = 600
    print_every = 1000
    lr = 1e-3
    lbfgs_steps = 1000
    steps = 100000
    outer_steps = 5
    decay_rate = 1
    os.makedirs(f'saved/{path}', exist_ok = True)
    t0 = time()
    best_criterion = 'loss'

    if train_model:
        params = (net_params, latent_params)
        opt_init, opt_update, get_params = optimizers.adam(lr)
        min_criterion = float('inf')
        best_params = params
        for i in range(outer_steps):
            args = [precompute_latent_fixed(d, phy_dim, Q_bd, Ai, Bi, jnp.linspace(0,t_terminal,1000)[:,None], h, c, path, restore = False)]
            params, cost, hmin, loss = train(i, params)
            old_eps, old_l = eps, l
            eps, l = 1e-5, 1e-4
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
    plot_swarm(params)
    t2 = time()
    print('Plot time: ' + str(t2 - t1))
