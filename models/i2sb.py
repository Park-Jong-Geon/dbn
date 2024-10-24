from functools import partial
from typing import Any, Callable, Sequence
import flax.linen as nn
import jax.numpy as jnp
import jax
import numpy as np

from .mlp_mixer import MlpMixer


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    # b, c, *spatial = y[0].shape
    b, *spatial, c = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += jnp.array([matmul_ops], dtype=jnp.double)


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""
    embedding_size: int = 256
    scale: float = 1.0

    @nn.compact
    def __call__(self, x):
        W = self.param(
            "W", jax.nn.initializers.normal(stddev=self.scale), (self.embedding_size,)
        )
        W = jax.lax.stop_gradient(W)
        x_proj = x[:, None] * W[None, :] * 2 * jnp.pi
        return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)
    

class ClsUnet(nn.Module):
    emb_dim: int = 256
    prob_embedding_channels: int = 4
    time_embedding_channels: int = 4
    imagefeature_embedding_channels: int = 24
    
    num_blocks: int = 32
    num_blocks_for_imagefeature_embedding: int = 32
    
    fourier_scale: float = 1.
    
    act:          Callable = nn.gelu
    fc:           nn.Module = partial(nn.Dense, use_bias=True,
                                      kernel_init=jax.nn.initializers.he_normal(),
                                      bias_init=jax.nn.initializers.zeros)
    
    def multichannel_fc(self, input, out_dim, ch):
        output = self.fc(out_dim * ch)(input)
        output = self.act(output)
        output = output.reshape(output.shape[:-1] + (out_dim, ch))
        return output
    
    @nn.compact
    def __call__(self, p, feat, t, **kwargs):
        # Probability embedding
        p_emb = self.multichannel_fc(p, self.emb_dim, self.prob_embedding_channels)
        
        # Timestep embedding; Gaussian Fourier features embeddings
        t_emb = GaussianFourierProjection(embedding_size=self.emb_dim//4, scale=self.fourier_scale)(t)
        t_emb = self.multichannel_fc(t_emb, self.emb_dim, self.time_embedding_channels)
        
        # Image features embedding; MLP-mixer block embeddings
        z_emb = MlpMixer(out_dim=self.emb_dim, 
                       num_blocks=self.num_blocks_for_imagefeature_embedding, 
                       hidden_dim=self.emb_dim, 
                       tokens_mlp_dim=self.emb_dim,
                       channels_mlp_dim=self.emb_dim)(feat)
        z_emb = self.multichannel_fc(z_emb, self.emb_dim, self.imagefeature_embedding_channels)

        p_emb = p_emb[..., None]
        emb = jnp.concatenate([z_emb, t_emb], axis=-1)
        emb = emb[..., None]
        h = MlpMixer(out_dim=p.shape[-1], 
                     num_blocks=self.num_blocks, 
                     hidden_dim=self.emb_dim, 
                     tokens_mlp_dim=self.emb_dim,
                     channels_mlp_dim=self.emb_dim)(p_emb, emb)
        return h
    

class DirichletFlowNetwork(nn.Module):
    base_net: nn.Module
    score_net: Sequence[nn.Module]
    cls_net: Sequence[nn.Module]
    crt_net: Sequence[nn.Module]
    max_t: float = None
    steps: float = None
    scale: float = 1.

    def setup(self):
        self.base = self.base_net()
        self.score = self.score_net()
        if self.cls_net is not None:
            self.cls = self.cls_net()
        if self.crt_net is not None:
            self.crt = self.crt_net()

    def encode(self, x, params_dict=None, **kwargs):
        # x: BxHxWxC
        if params_dict is not None:
            out = self.base.apply(params_dict, x, **kwargs)
        out = self.base(x, **kwargs)
        return out

    def correct(self, z, **kwargs):
        if self.crt_net is None:
            return z
        return self.crt(z)

    def classify(self, z, params_dict=None, **kwargs):
        def _classify(cls, _z):
            if params_dict is not None:
                return cls.apply(params_dict, _z, **kwargs)
            return cls(_z, **kwargs)
        out = _classify(self.cls, z)
        return out

    def __call__(self, *args, **kwargs):
        return self.conditional_dbn(*args, **kwargs)

    def conditional_dbn(self, rng, l_label, x, base_params=None, cls_params=None, **kwargs):
        z = self.encode(x, base_params, **kwargs)
        self.classify(z, cls_params, **kwargs)
        x_t, t, next_x_t = self.forward(rng, l_label)
        eps = self.score(x_t, z, t, **kwargs)
        return eps, next_x_t

    def forward(self, rng, l_label, t=None):
        B = l_label.shape[0]
        C = l_label.shape[1]
        
        # Sample t
        t_rng, n_rng, d_rng = jax.random.split(rng, 3)
        if t is None:
            t = jax.random.uniform(t_rng, (l_label.shape[0],), maxval=self.max_t)  # (B,)
        
        # Sample classes
        p_ens = jax.nn.softmax(l_label, axis=-1)
        cls = jax.random.categorical(n_rng, p_ens)
        e_cls = jax.nn.one_hot(cls, C)
        alpha = 1 + t[..., None] * e_cls
        
        # Sample x_t
        x_t = jax.random.dirichlet(d_rng, alpha)
        x_t = x_t + x_t[jnp.arange(B), cls][..., None] * (p_ens - e_cls)
        
        # Evaluate u_t
        u_t = self.u_t(x_t, cls, t, p_ens)
        next_x_t = x_t + (self.max_t / self.steps) * u_t
        
        return x_t, t, next_x_t

    def grad_betainc(self, a, b, x, h=1e-3):
        f1 = jax.scipy.special.betainc(a+2*h, b, x)
        f2 = jax.scipy.special.betainc(a+h, b, x)
        f3 = jax.scipy.special.betainc(a-h, b, x)
        f4 = jax.scipy.special.betainc(a-2*h, b, x)
        return (-f1+8*f2-8*f3+f4) / (12*h)

    def u_t(self, x_t, cls, t, target):        
        B = x_t.shape[0]
        K = x_t.shape[-1]
        log_b = jax.scipy.special.betaln(t+1, K-1)
        
        y_j = x_t[jnp.arange(B), cls] / target[jnp.arange(B), cls]
        
        I = jax.vmap(self.grad_betainc, in_axes=(0, None, 0))(t+1, K-1, y_j)
        C = - I * jnp.exp(log_b - (K-1) * jnp.log(1 - y_j) - t * jnp.log(y_j))
        C = jnp.nan_to_num(C)
        
        return C[..., None] * (target - x_t)

    def sample(self, *args, **kwargs):
        return self.conditional_sample(*args, **kwargs)

    def _score(self, x_t, z, t, **kwargs):
        return self.score(x_t, z, t, **kwargs) / self.scale

    def conditional_sample(self, rng, sampler, x):
        zB = self.encode(x, training=False)
        lB = self.classify(zB, training=False)
        _lB = jax.random.dirichlet(rng, jnp.ones(lB.shape[1]), (lB.shape[0],))
        lC = sampler(
            partial(self._score, training=False), rng, _lB, zB)
        lC = lC[None, ...]
        return lC, lB
