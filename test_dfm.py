import jax
import jax.numpy as jnp
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import math

def batch_mul(a, b):
    return jax.vmap(lambda a, b: a * b)(a, b)

# def pt_ut_per_class(x_t, t, K, h=1e-3):
#     @jax.jit
#     def grad_betainc(a, b, x):
#         f1 = jax.scipy.special.betainc(a+2*h, b, x)
#         f2 = jax.scipy.special.betainc(a+h, b, x)
#         f3 = jax.scipy.special.betainc(a-h, b, x)
#         f4 = jax.scipy.special.betainc(a-2*h, b, x)
#         return (-f1+8*f2-8*f3+f4) / (12*h)
    
#     gamma = jnp.exp(jax.scipy.special.gammaln(K-1))
#     I = jax.vmap(grad_betainc, in_axes=(0, None, 0))(t+1, K-1, x_t)
#     pt_C = - gamma * I / (1-x_t)**(K-1)
#     pt_C = jnp.nan_to_num(pt_C)

#     vec = jnp.repeat(jnp.eye(K)[None, ...], x_t.shape[0], axis=0) - x_t[None, ...]
#     return vec * pt_C[..., None]

# def u_t(x_t, t, p_ens):
#     num_classes = p_ens.shape[-1]
        
#     log_p_t = []
#     eye = jnp.eye(num_classes)
#     for e in eye:
#         alpha = 1 + t[..., None] * e
#         log_p_t.append(jax.vmap(jax.scipy.stats.dirichlet.logpdf, in_axes=(0, 0))(x_t, alpha))
#     log_p_t = jnp.stack(log_p_t, axis=-1)
    
#     sum_p = jnp.exp(jax.scipy.special.logsumexp(jnp.log(p_ens) + log_p_t))
#     u_t = jnp.sum(pt_ut_per_class(x_t, t, num_classes) * p_ens[..., None], axis=1) / sum_p[..., None]
#     print(u_t)
#     return u_t

def u_t_per_class(x_t, t, K, h=1e-3):
    @jax.jit
    def grad_betainc(a, b, x):
        f1 = jax.scipy.special.betainc(a+2*h, b, x)
        f2 = jax.scipy.special.betainc(a+h, b, x)
        f3 = jax.scipy.special.betainc(a-h, b, x)
        f4 = jax.scipy.special.betainc(a-2*h, b, x)
        return (-f1+8*f2-8*f3+f4) / (12*h)
    
    log_b = jax.scipy.special.betaln(t+1, K-1)
    I = jax.vmap(grad_betainc, in_axes=(0, None, 0))(t+1, K-1, x_t)
    C = - I * jnp.exp(log_b[..., None] - (K-1) * jnp.log(1 - x_t) - t[..., None] * jnp.log(x_t))
    C = jnp.nan_to_num(C)
    vec = jnp.repeat(jnp.eye(K)[None, ...], x_t.shape[0], axis=0) - jnp.expand_dims(x_t, 1)
    
    # print(I / (1 - x_t)**(K-1))
    return vec * C[..., None], C

def u_t(x_t, t, p_ens):
    num_classes = p_ens.shape[-1]
        
    log_p_t = []
    eye = jnp.eye(num_classes)
    for e in eye:
        alpha = 1 + t[..., None] * e
        log_p_t.append(jax.vmap(jax.scipy.stats.dirichlet.logpdf, in_axes=(0, 0))(x_t, alpha))
    log_p_t = jnp.stack(log_p_t, axis=-1)
    
    log_p_ens = jnp.log(p_ens)
    log_p = log_p_ens + log_p_t
    log_p = log_p - jax.scipy.special.logsumexp(log_p, -1, keepdims=True)
    p = jnp.exp(log_p)
    u_t_i, C = u_t_per_class(x_t, t, num_classes)
    u_t = jnp.sum(u_t_i * p[..., None], axis=1)
    # print(- C * jnp.exp(log_p_t))
    return u_t

def sample(rng, target, T, steps, B, C):
    timesteps = jnp.linspace(0, T, steps+1)
    
    @jax.jit
    def body_fn(n, x_n):
        current_t = jnp.array([timesteps[n]])
        next_t = jnp.array([timesteps[n+1]])
        current_t = jnp.tile(current_t, [B])
        next_t = jnp.tile(next_t, [B])

        eps = u_t(x_n, current_t, target)
        x_n = x_n + batch_mul(next_t - current_t, eps)
        x_n = jnp.where(x_n < 1e-8, 1e-8, x_n)
        x_n = x_n / jnp.sum(x_n, axis=-1, keepdims=True)
        return x_n
    
    x0 = jax.random.dirichlet(rng, jnp.ones(C), (B,))
    x_list = [jnp.array([x0])]
    val = x0
    for i in tqdm(range(0, steps)):
        val = body_fn(i, val)
        x_list.append(jnp.array([val]))

    return jnp.concatenate(x_list, axis=0)

def convert_coordinate(p):
    x = (p[1] - p[0]) / math.sqrt(2)
    y = math.sqrt(jnp.linalg.norm(p - jnp.array([0.5, 0.5, 0]))**2 - jnp.sum(x**2) + 1e-6)
    return x, y

if __name__ == '__main__':
    np.set_printoptions(edgeitems=10)
    T = 54
    steps = 200
    B = 10000
    C = 10
    rng = jax.random.PRNGKey(1568)
    rng, new_rng = jax.random.split(rng)
    # target = jax.random.dirichlet(rng, jnp.ones(C), (1,))
    # target = jax.random.dirichlet(rng, jnp.ones(C), (B,))
    target = jax.nn.softmax(jnp.array([[10, 2, 3, -1, 5, 4, 0, 6, 3, 3]]))
    samples = sample(new_rng, target, T, steps, B, C)
    # print(sample(new_rng, target, T, steps, B, C))
    
    # x, y = convert_coordinate(target[0])
    # traj_x = [x, -1/math.sqrt(2), 1/math.sqrt(2), 0]
    # traj_y = [y, 0, 0, math.sqrt(3)/math.sqrt(2)]
    # for p in samples:
    #     x, y = convert_coordinate(p)
    #     traj_x.append(x)
    #     traj_y.append(y)
    # print(f'source: {samples[0]}')
    print(f'target: {target}')
    print(f'matched: {samples[-1].mean(0)}')
    # plt.scatter(traj_x, traj_y)
    # plt.savefig('debug.png')