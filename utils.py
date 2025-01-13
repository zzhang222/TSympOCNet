import jax.numpy as jnp

def relu(x):
    return jnp.maximum(0, x)

def mse(x, y):
    return ((x - y) ** 2).sum() / x.size