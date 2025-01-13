import matplotlib.pyplot as plt
from net_time import predict_qp_from_Hamiltonian
import pickle
import jax.numpy as jnp
from example_time import init_bd_simple
from net_time import get_latent_pre, t_sympnet, precompute_latent_fixed
import os

def dist_to_point(x):
    return jnp.sqrt(1e-10+jnp.sum((x.reshape([-1,1,phy_dim]) - ws[None,...])**2, axis = -1)) - (dr + qr)

def h(q):
    x, v = q[:d], q[d:]
    h_obs = dist_to_point(x)
    h_obs = h_obs.flatten()
    # avoid hitting between drones
    x = x.reshape([d // phy_dim, phy_dim, 1])
    x = jnp.transpose(x, (1, 0, 2))
    y = jnp.transpose(x, (0, 2, 1))
    z = jnp.sum((x - y)**2, axis = 0)
    n = d // phy_dim
    min_value = z.flatten()[1:].reshape([n-1, n+1])[:,:-1]
    min_value = jnp.sqrt(min_value.flatten() + 1e-10) - (dr * 2)
    h = jnp.concatenate([h_obs, min_value], axis = 0)
    return h

def plot_traj():
    fig = plt.figure(figsize = [15,6])
    qs = []
    qs_old = []
    for seed in range(1, 6):
        path = f'obs_TSympNet_{seed}_8_{Cv}_{k}_{c}'
        q, _ = pickle.load(open(f'saved/predicted_qp_{path}', "rb"))
        q_old, _ = pickle.load(open(f'saved/predicted_qp_{path}_old', "rb"))
        qs.append(q)
        qs_old.append(q_old)

    x = jnp.linspace(-halfroomsz, halfroomsz, 100)
    y = jnp.linspace(-halfroomsz, halfroomsz, 100)
    xx, yy = jnp.meshgrid(x, y)
    pp = jnp.concatenate([xx.reshape([-1,1]), yy.reshape([-1,1])], axis = -1)
    for i in range(5):
        ax = plt.subplot(2,5,i+1)
        plt.title(f'Test {i+1}', fontsize = 15)
        plt.xlim((-halfroomsz*1.2, halfroomsz*1.2))
        plt.ylim((-halfroomsz*1.2, halfroomsz*1.2))
        if i == 0:
            plt.ylabel(r'Update $v_0$', fontsize = 15)
        if num_obs != 0:
            zz = dist_to_point(pp) + dr
            zz = jnp.min(zz, axis = 1).reshape([100, 100])
            plt.contour(xx,yy,zz,[0])
        for j in range(4):
            patch = plt.Circle((qs[i][-1, 2*j], qs[i][-1, 2*j+1]), dr, fill = False, color = 'black')
            ax.add_patch(patch)
            plt.plot(qs[i][:, 2*j], qs[i][:, 2*j+1], c = 'gray', alpha = 0.2)

        ax = plt.subplot(2,5,i+6)
        plt.xlim((-halfroomsz*1.2, halfroomsz*1.2))
        plt.ylim((-halfroomsz*1.2, halfroomsz*1.2))
        if num_obs != 0:
            zz = dist_to_point(pp) + dr
            zz = jnp.min(zz, axis = 1).reshape([100, 100])
            plt.contour(xx,yy,zz,[0])
        if i == 0:
            plt.ylabel('No update', fontsize = 15)
        for j in range(4):
            patch = plt.Circle((qs_old[i][-1, 2*j], qs_old[i][-1, 2*j+1]), dr, fill = False, color = 'black')
            ax.add_patch(patch)
            plt.plot(qs_old[i][:, 2*j], qs_old[i][:, 2*j+1], c = 'gray', alpha = 0.2)
    plt.tight_layout()
    plt.savefig('figs/obs2.pdf')

def plot_t():
    fig = plt.figure(figsize = [12,6])
    net_params, latent_params = params_single
    if os.path.exists(f'saved/ground_truth_{path}'):
        q2, p2 = pickle.load(open(f'saved/ground_truth_{path}', "rb"))
    else:
        q2, p2 = predict_qp_from_Hamiltonian(act, transform, get_latent, net_params, latent_params, t, A, B, phy_dim, d, eps, l, h, Q_bd, m, k, c, True, *args)
        pickle.dump((q2, p2), open(f'saved/ground_truth_{path}', "wb"))
    
    q, p = pickle.load(open(f'saved/predicted_qp_{path_all}', "rb"))
    q, v = q[:, :8], q[:, 8:]
    u = p[:, 8:] / m      
    q2, v2 = q2[:, :d], q2[:, d:]
    u2 = p2[:, d:] / m
    rotate = jnp.block([[0,-1], [1,0]])

    for i in range(4):
        plt.subplot(2, 4, i+1)
        plt.plot(t, q[:, i*2], 'k', label = r'$x$, NN')
        plt.plot(t, v[:, i*2], 'r', label = r'$v_x$, NN')
        plt.plot(t, u[:, i*2], 'b', label = r'$u_x$, NN')
        plt.plot(t, q2[:, 0], 'k--', label = r'$x$, shooting')
        plt.plot(t, v2[:, 0], 'r--', label = r'$v_y$, shooting')
        plt.plot(t, u2[:, 0], 'b--', label = r'$u_y$, shooting')
        if i == 0:
            plt.legend()
        plt.title(f'Agent {i+1}')
        plt.subplot(2, 4, i+5)
        plt.plot(t, q[:, 2*i+1], 'k', label = r'$y$, NN')
        plt.plot(t, v[:, 2*i+1], 'r', label = r'$v_y$, NN')
        plt.plot(t, u[:, 2*i+1], 'b', label = r'$u_y$, NN')
        plt.plot(t, q2[:, 1], 'k--', label = r'$y$, shooting')
        plt.plot(t, v2[:, 1], 'r--', label = r'$v_y$, shooting')
        plt.plot(t, u2[:, 1], 'b--', label = r'$u_y$, shooting')
        q2 @= rotate
        v2 @= rotate
        u2 @= rotate
        if i == 0:
            plt.legend()
        plt.xlabel(r'$t$', fontsize = 15)
    plt.tight_layout()
    plt.savefig(f'figs/obs2_t.pdf')
    
if __name__ == '__main__':
    act = jnp.tanh
    get_latent = get_latent_pre
    transform = t_sympnet
    t_terminal = 10
    num_t = 100
    t = jnp.linspace(0, t_terminal, num_t)[:, None]
    Q_bd = init_bd_simple(None, None)
    d = 2
    m = 1
    phy_dim = 2
    halfroomsz = 0.5
    dr = 0.05
    params = []
    ws = jnp.array([[0,0]])
    num_obs = len(ws)
    qr = 0.3 * halfroomsz / len(ws)
    ts = [0.0, 3.3, 6.7, 10.0]
    Cv = 0.5
    A = jnp.concatenate([jnp.zeros([d, d]), jnp.eye(d, d)], axis = 1)
    A = jnp.concatenate([A, jnp.zeros_like(A)], axis = 0)
    B = jnp.concatenate([jnp.zeros([d, d]), jnp.eye(d, d)], axis = 0)
    k = 0.0
    c = 1.0
    path = f'obs_TSympNet_-1_{d}_{Cv}_{k}_{c}'
    path_all = f'obs_TSympNet_4_8_{Cv}_{k}_{c}'
    params_single = pickle.load(open(f'saved/last_params_{path}', "rb"))
    l = eps = 1e-3
    args = [precompute_latent_fixed(d, phy_dim, Q_bd, A, B, jnp.linspace(0,t_terminal,1000)[:,None], h, m, c, path, restore = False)]
    #plot_t()
    plot_traj()