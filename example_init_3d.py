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
    return jnp.array([q0, qT]).reshape([2,-1])

def h(x):
    def compute_single_barrier(barrier):
        h1 = jnp.maximum(x_new[:, 0] - barrier[0], -x_new[:, 0] + barrier[1])
        h2 = jnp.maximum(x_new[:, 1] - barrier[2], -x_new[:, 1] + barrier[3])
        h3 = jnp.maximum(x_new[:, 2] - barrier[4], -x_new[:, 2] + barrier[5])
        return jnp.max(jnp.stack([h1, h2, h3]), axis=0)
    x_new = x.reshape([d // phy_dim, phy_dim])
    # avoid hitting between drones
    h = vmap(compute_single_barrier)(barriers)
    h = jnp.concatenate(h, axis=0)
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
    from mpl_toolkits.mplot3d import Axes3D
    net_params, latent_params = params
    q, v = predict_qp(transform, get_latent, net_params, latent_params, t, t_terminal, *args)
    q, v = np.array(q), np.array(v)
    v0, vT = v[0], v[-1]
    pickle.dump((v0, vT), open(f'saved/init_v_{path}', "wb"))
    num_drone = d // phy_dim
    q_traj = q.reshape([-1,num_t,phy_dim])
    qT = bdy[1].reshape([-1,phy_dim])
    num_drone = len(qT)
    #num_drone = net.dim // 2
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
    anim.save('figs/init_3d.gif', writer='imagemagick', fps=30)
    plt.show()
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SympOCnet')
    parser.add_argument('--net_type', type = str, default = 'TSympNet', help = 'network type')
    parser.add_argument('--seed', default=-1, type=int, help='random seed')

    args = parser.parse_args()
    os.makedirs('saved', exist_ok = True)
    os.makedirs('figs', exist_ok = True)
    key = random.PRNGKey(args.seed)
    phy_dim = 3
    num_t = 200
    t_terminal = 10
    t = jnp.linspace(0, t_terminal, num_t)[:, None]
    barriers = jnp.array([[2, -2, 0.5, -0.5, 7, 0], [4, 2, 1, -1, 4, 0]])
    bdy = init_bd()
    d = bdy.shape[1]
    dr = 0.2
    eps = 1e-5
    l = 1e-4
    lam = 600

    path = f'{args.net_type}_{args.seed}_{d}'
    os.makedirs(f'saved/{path}', exist_ok = True)
    lr = 1e-3
    steps = 300000
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
