import matplotlib.pyplot as plt
from net_time import *
import jax.numpy as jnp
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches

def make_legend_arrow(legend, orig_handle,
                      xdescent, ydescent,
                      width, height, fontsize):
    p = mpatches.FancyArrow(0, 0.5*height, width, 0, length_includes_head=True, head_width=0.75*height )
    return p

def dist(x):
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
    h1 = x_new[:,0] + halfroomsz             # x >= -C
    h2 = -x_new[:,0] + halfroomsz          # x <= C
    h3 = x_new[:,1] + halfroomsz           # y >= -C
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

def init_bd(v0, vT, numwall, numdperedge):
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

def plot_plane():
    fig = plt.figure(figsize = [16,12])
    x = np.linspace(-halfroomsz, halfroomsz, 100)
    y = np.linspace(-halfroomsz, halfroomsz, 100)
    xx, yy = np.meshgrid(x, y)
    p = np.concatenate([xx.reshape([-1,1]), yy.reshape([-1,1])], axis = -1)
    zz = dist(p) + dr
    zz = np.min(zz, axis = 1).reshape([100, 100])

    for i in range(num_d):
        net_params, latent_params = params[i]
        q, p = predict_qp(act, transform, get_latent, net_params, latent_params, t, t_terminal, qp[i])
        q, v = q[:, :d], q[:, d:]
        u = p[:, d:]
        num_drone = d // phy_dim
        
        for j in range(num_t):
            ax = plt.subplot(num_d, num_t, i * num_t + j + 1)
            plt.contour(xx,yy,zz,[0])
            if i == 0:
                plt.title(f't = {ts[j]}', fontsize = 18)
            if j == 0:
                plt.ylabel(f'Trial {i+1}', fontsize = 18)
            s = 99 // (num_t - 1) * j
            plt.xlim((-halfroomsz*1.2, halfroomsz*1.2))
            plt.ylim((-halfroomsz*1.2, halfroomsz*1.2))
            
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
                plt.plot(q[:s+1, 2*k], q[:s+1, 2*k+1], c = 'gray', alpha = 0.2)
            if i == 0 and j == 0:
                plt.legend([arrow_v, arrow_u], [label_v, label_u], handler_map={mpatches.FancyArrow : HandlerPatch(patch_func=make_legend_arrow)})

    plt.tight_layout()
    plt.savefig(f'figs/obs.pdf')

if __name__ == '__main__':
    act = lambda x: x * nn.sigmoid(x)
    get_latent = get_latent_pre
    transform = t_sympnet
    t_terminal = 10
    num_t = 100
    halfroomsz = 0.5
    Q_bd = init_bd(None, None, 1, 1)
    t = jnp.linspace(0, t_terminal, num_t)[:, None]
    phy_dim = 2
    Ai = jnp.concatenate([jnp.zeros([phy_dim, phy_dim]), jnp.eye(phy_dim, phy_dim)], axis = 1)
    Ai = jnp.concatenate([Ai, jnp.zeros_like(Ai)], axis = 0)
    Bi = jnp.concatenate([jnp.zeros([phy_dim, phy_dim]), jnp.eye(phy_dim, phy_dim)], axis = 0)
    dr = 0.02
    params = []
    ts = [0.0, 3.3, 6.7, 10.0]
    Cv = 0.5
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
    seeds = [4,5,7]
    qp = []
    d = 8
    for seed in seeds:
        path = f'maze_TSympNet_{seed}_{d}_{Cv}_0.0_0.0'
        params.append(pickle.load(open(f'saved/last_params_{path}', "rb")))
        qp.append(precompute_latent_fixed(d, 2, Q_bd, Ai, Bi, jnp.linspace(0,t_terminal,1000)[:,None], h, 0, path, restore = False))
    num_d = len(seeds)
    num_t = len(ts)
    plot_plane()