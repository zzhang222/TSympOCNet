from jax import grad, jit
from net import Hamiltonian
from scipy import optimize

def dynamics(q0, p0, qT, A, B, t, dt, phy_dim, d, eps, l, h):
    q = q0
    p = p0
    for i in range(len(t) - 1):
        dH = grad(Hamiltonian, argnums=(0, 1))(q, p, phy_dim, d, eps, l, h, A, B)
        q = q + dt * dH[1]
        p = p - dt * dH[0]
    return q - qT

def shooting(bdy, p0, A, B, t, phy_dim, d, eps, l, h):
    q0, qT = bdy
    dt = t[1] - t[0]
    objective = lambda p0: dynamics(q0, p0, qT, A, B, t, dt, phy_dim, d, eps, l, h)
    res = optimize.fsolve(objective, p0)
    return res.x


