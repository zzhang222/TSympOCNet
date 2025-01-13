from net_time import *
import matplotlib.pyplot as plt
import jax.numpy as jnp
from example_time import init_bd_simple
import pickle
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches

def dist_to_point(x):
    return jnp.sqrt(1e-10+jnp.sum((x.reshape([-1,1,phy_dim]) - ws[None,...])**2, axis = -1)) - (dr + qr)

def make_legend_arrow(legend, orig_handle,
                      xdescent, ydescent,
                      width, height, fontsize):
    p = mpatches.FancyArrow(0, 0.5*height, width, 0, length_includes_head=True, head_width=0.75*height )
    return p

def h(q):
    x, v = q[:d], q[d:]
    x_new = x.reshape([d // phy_dim, phy_dim])
    h1 = x_new[:,0] + halfroomsz * 1.2             # x >= -C
    h2 = -x_new[:,0] + halfroomsz * 1.2          # x <= C
    h3 = x_new[:,1] + halfroomsz  * 1.2           # y >= -C
    h4 = -x_new[:,1] + halfroomsz * 1.2           # y <= C
    h_bd = jnp.concatenate([h1, h2, h3, h4], axis = 0)
    h_obs = dist_to_point(x)
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
    h = jnp.concatenate([hv, h_obs, min_value], axis = 0)
    return h

def plot_t():
    fig = plt.figure(figsize = [12,8])
    for i in range(num_k):
        net_params, latent_params = params[i]
        q, p = predict_qp(act, transform, get_latent, net_params, latent_params, t, t_terminal, Q_bd, m)
        k = ks[i]
        q2, p2 = predict_qp_from_Hamiltonian(act, transform, get_latent, net_params, latent_params, t, A, B, phy_dim, d, eps, l, h, Q_bd, m, k, tune_with_shooting)
        #q2, p2 = q, p
        print(q2)
        q, v = q[:, :d], q[:, d:]
        u = p[:, d:] / m
        q2, v2 = q2[:, :d], q2[:, d:]
        u2 = p2[:, d:] / m
        plt.subplot(2, num_k, i + 1)
        plt.title(rf'$k = {ks[i]}$', fontsize = 18)
        if i == 0:
            plt.plot(t, q[:, 0], 'k', label = r'$x$, NN')
            plt.plot(t, v[:, 0], 'r', label = r'$v_x$, NN')
            plt.plot(t, u[:, 0], 'b', label = r'$u_x$, NN')
            plt.plot(t, q2[:, 0], 'k--', label = r'$x$, shooting')
            plt.plot(t, v2[:, 0], 'r--', label = r'$v_y$, shooting')
            plt.plot(t, u2[:, 0], 'b--', label = r'$u_y$, shooting')
            plt.legend()
        else:
            plt.plot(t, q[:, 0], 'k')
            plt.plot(t, v[:, 0], 'r')
            plt.plot(t, u[:, 0], 'b')
            plt.plot(t, q2[:, 0], 'k--')
            plt.plot(t, v2[:, 0], 'r--')
            plt.plot(t, u2[:, 0], 'b--')
        plt.subplot(2, num_k, i + 1 + num_k)
        if i == 0:
            plt.plot(t, q[:, 1], 'k', label = r'$y$, NN')
            plt.plot(t, v[:, 1], 'r', label = r'$v_y$, NN')
            plt.plot(t, u[:, 1], 'b', label = r'$u_y$, NN')
            plt.plot(t, q2[:, 1], 'k--', label = r'$y$, shooting')
            plt.plot(t, v2[:, 1], 'r--', label = r'$v_y$, shooting')
            plt.plot(t, u2[:, 1], 'b--', label = r'$u_y$, shooting')
            plt.legend()
            plt.ylabel(r'$x$', fontsize = 15)
        else:
            plt.plot(t, q[:, 1], 'k')
            plt.plot(t, v[:, 1], 'r')
            plt.plot(t, u[:, 1], 'b')
            plt.plot(t, q2[:, 1], 'k--')
            plt.plot(t, v2[:, 1], 'r--')
            plt.plot(t, u2[:, 1], 'b--')
        plt.xlabel(r'$t$', fontsize = 15)
    plt.tight_layout()
    plt.savefig(f'figs/obs_t.pdf')

def plot_plane():
    fig = plt.figure(figsize = [16,12])
    for i in range(num_k):
        net_params, latent_params = params[i]
        q, p = predict_qp(act, transform, get_latent, net_params, latent_params, t, t_terminal, Q_bd, m)
        #q, p = predict_qp_from_sympnet(act, transform, get_latent, net_params, latent_params, t, t_terminal, Q_bd, m)
        #q, p = predict_qp_from_dynamics(act, transform, get_latent, net_params, latent_params, t, t_terminal, A, B, Q_bd, m, k)
        #q, p = predict_qp_from_Hamiltonian(act, transform, get_latent, net_params, latent_params, Q_bd, t, A, B, phy_dim, d, eps, l, h, Q_bd, m, k, tune_with_shooting)
        q, v = q[:, :d], q[:, d:]
        u = p[:, d:] / m
        num_drone = d // phy_dim
        
        x = jnp.linspace(-halfroomsz, halfroomsz, 100)
        y = jnp.linspace(-halfroomsz, halfroomsz, 100)
        xx, yy = jnp.meshgrid(x, y)
        p = jnp.concatenate([xx.reshape([-1,1]), yy.reshape([-1,1])], axis = -1)
        for j in range(num_t):
            ax = plt.subplot(num_k, num_t, i * num_t + j + 1)
            if i == 0:
                plt.title(f't = {ts[j]}', fontsize = 18)
            if j == 0:
                plt.ylabel(f'k = {ks[i]}', fontsize = 18)
            s = 99 // (num_t - 1) * j
            plt.xlim((-halfroomsz*1.2, halfroomsz*1.2))
            plt.ylim((-halfroomsz*1.2, halfroomsz*1.2))
            if num_obs != 0:
                zz = dist_to_point(p) + dr
                zz = jnp.min(zz, axis = 1).reshape([100, 100])
                plt.contour(xx,yy,zz,[0])
            if i == 0 and j == 0:
                label_q = 'x'
                label_v = 'v'
                label_u = 'u'
            else:
                label_q = label_v = label_u = None
            for k in range(num_drone):
                patch = plt.Circle((q[s, 2*k], q[s, 2*k+1]), dr, fill = False, color = 'black')
                ax.add_patch(patch)
                arrow_v = plt.arrow(q[s, 2*k], q[s, 2*k+1], v[s, 2*k], v[s, 2*k+1], width = 0.01, fill = False, color = 'red')
                arrow_u = plt.arrow(q[s, 2*k], q[s, 2*k+1], u[s, 2*k], u[s, 2*k+1], width = 0.01, fill = False, color = 'blue')
                plt.plot(q[:s+1, 2*k], q[:s+1, 2*k+1], c = 'gray', alpha = 0.1)
            if i == 0 and j == 0:
                plt.legend([arrow_v, arrow_u], [label_v, label_u], handler_map={mpatches.FancyArrow : HandlerPatch(patch_func=make_legend_arrow)})

    plt.tight_layout()
    plt.savefig(f'figs/obs.pdf')

if __name__ == '__main__':
    act = jnp.tanh
    get_latent = get_latent_fixed
    transform = t_sympnet
    t_terminal = 10
    num_t = 100
    t = jnp.linspace(0, t_terminal, num_t)[:, None]
    Q_bd = init_bd_simple()
    d = 2
    m = 1
    phy_dim = 2
    halfroomsz = 0.5
    dr = 0.05
    params = []
    ws = jnp.array([[0,0]])
    num_obs = len(ws)
    qr = 0.3 * halfroomsz / len(ws)
    ks = [0.0, 1.0, 2.0]
    ts = [0.0, 3.3, 6.7, 10.0]
    Cv = 0.5
    A = jnp.concatenate([jnp.zeros([d, d]), jnp.eye(d, d)], axis = 1)
    A = jnp.concatenate([A, jnp.zeros_like(A)], axis = 0)
    B = jnp.concatenate([jnp.zeros([d, d]), jnp.eye(d, d)], axis = 0)
    for k in ks:
        path = f'obs_TSympNet_-1_{d}_{Cv}_{k}'
        params.append(pickle.load(open(f'saved/last_params_{path}', "rb")))
    num_k = len(ks)
    num_t = len(ts)
    eps = 1e-3
    l = 1e-3
    outer_steps = 10
    decay_rate = 0.8
    tune_with_shooting = True#False
    #plot_plane()
    plot_t()