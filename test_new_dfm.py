import jax
import jax.numpy as jnp
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import math

def batch_mul(a, b):
    return jax.vmap(lambda a, b: a * b)(a, b)

# def u_t(x_t, cls, t, target, h=1e-3):
#     @jax.jit
#     def grad_betainc(a, b, x):
#         f1 = jax.scipy.special.betainc(a+2*h, b, x)
#         f2 = jax.scipy.special.betainc(a+h, b, x)
#         f3 = jax.scipy.special.betainc(a-h, b, x)
#         f4 = jax.scipy.special.betainc(a-2*h, b, x)
#         return (-f1+8*f2-8*f3+f4) / (12*h)
    
#     K = x_t.shape[-1]
    
#     log_b = jax.scipy.special.betaln(t+1, K-1)
    
#     idx = jnp.stack([jnp.arange(x_t.shape[0]), cls], axis=-1)
#     x_j = jax.vmap(lambda i: x_t[tuple(i)])(idx)
#     target_j = jax.vmap(lambda i: target[tuple(i)])(idx)
    
#     I = jax.vmap(grad_betainc, in_axes=(0, None, 0))(t+1, K-1, x_j)
    
#     C = - I * jnp.exp(log_b - (K-1) * jnp.log(1 - x_j) - t * jnp.log(x_j))
#     C = C / x_j # Magic number?
#     C = jnp.nan_to_num(C)
#     return C[..., None] * (target - x_t)

def u_t(x_t, cls, t, target, h=1e-3):
    @jax.jit
    def grad_betainc(a, b, x):
        f1 = jax.scipy.special.betainc(a+2*h, b, x)
        f2 = jax.scipy.special.betainc(a+h, b, x)
        f3 = jax.scipy.special.betainc(a-h, b, x)
        f4 = jax.scipy.special.betainc(a-2*h, b, x)
        return (-f1+8*f2-8*f3+f4) / (12*h)
    
    K = x_t.shape[-1]
    
    log_b = jax.scipy.special.betaln(t+1, K-1)
    
    idx = jnp.stack([jnp.arange(x_t.shape[0]), cls], axis=-1)    
    x_j = jax.vmap(lambda i: x_t[tuple(i)])(idx)
    target_j = jax.vmap(lambda i: target[tuple(i)])(idx)
    
    
    y_j = x_j / target_j
    I = jax.vmap(grad_betainc, in_axes=(0, None, 0))(t+1, K-1, y_j)
    
    C = - I * jnp.exp(log_b - (K-1) * jnp.log(1 - y_j) - t * jnp.log(y_j))
    C = jnp.nan_to_num(C)
    return C[..., None] * (target - x_t)

def sample(source, target, cls, T, steps):
    timesteps = jnp.linspace(0, T, steps+1)
    
    @jax.jit
    def body_fn(n, x_n):
        current_t = jnp.array([timesteps[n]])
        next_t = jnp.array([timesteps[n+1]])
        current_t = jnp.tile(current_t, [B])
        next_t = jnp.tile(next_t, [B])

        eps = u_t(x_n, cls, current_t, target)
        x_n = x_n + batch_mul(next_t - current_t, eps)
        return x_n
    
    x_list = [jnp.array(source)]
    val = source
    for i in tqdm(range(0, steps)):
        val = body_fn(i, val)
        x_list.append(jnp.array(val))

    return jnp.concatenate(x_list, axis=0)

def source(rng, target, B, C):
    cls = jax.random.categorical(rng, jnp.ones((B, C)))
    e_i = jax.nn.one_hot(cls, C)
    x = jax.random.dirichlet(rng, jnp.ones(C), (B,))
    
    idx = jnp.stack([jnp.arange(B), cls], axis=-1)
    x_cls = jax.vmap(lambda i: x[tuple(i)])(idx)
    
    return x + x_cls[..., None] * (target - e_i), cls

def convert_coordinate(p):
    x = (p[1] - p[0]) / math.sqrt(2)
    y = math.sqrt(jnp.linalg.norm(p - jnp.array([0.5, 0.5, 0]))**2 - jnp.sum(x**2) + 1e-6)
    return x, y

if __name__ == '__main__':
    np.set_printoptions(edgeitems=10)
    T = 200
    steps = 1000
    B = 1000
    C = 10
    
    rng = jax.random.PRNGKey(14)
    rng, new_rng = jax.random.split(rng)

    # target = jax.random.dirichlet(rng, jnp.ones(C), (B,))
    target = jax.nn.softmax(jnp.array([6, 4, -2, 1, 0, 6, 3, -5, 2, 1]))
    target = jnp.tile(target, [B, 1])
    _source, cls = source(new_rng, target, B, C)
    samples = sample(_source, target, cls, T, steps)
    
    print(f'source: {samples[0:B]}')
    print(f'target: {target}')
    print(f'matched: {samples[-B:]}')
    
    print(jnp.mean(jnp.linalg.norm(samples[-B:] - target, axis=-1)))
    print(jnp.var(jnp.linalg.norm(samples[-B:] - target, axis=-1)))
    
    # x, y = convert_coordinate(target[0])
    # traj_x = [x, -1/math.sqrt(2), 1/math.sqrt(2), 0]
    # traj_y = [y, 0, 0, math.sqrt(3)/math.sqrt(2)]
    # for p in samples:
    #     x, y = convert_coordinate(p)
    #     traj_x.append(x)
    #     traj_y.append(y)
    
    # plt.scatter(traj_x, traj_y)
    # plt.savefig('debug.png')