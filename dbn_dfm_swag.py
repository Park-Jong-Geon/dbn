from builtins import NotImplementedError
import os
import orbax
from easydict import EasyDict


from typing import Any

import flax
from flax.training import train_state, common_utils, checkpoints
from flax.training import dynamic_scale as dynamic_scale_lib
from flax.core.frozen_dict import freeze
from flax import jax_utils

import optax

import jax.numpy as jnp
import numpy as np
import jax
import jaxlib

import datetime
import wandb

import defaults_dsb as defaults
from tabulate import tabulate
import sys
from data.build import build_dataloaders 
from giung2.metrics import evaluate_acc, evaluate_nll
from giung2.models.layers import FilterResponseNorm
from models.resnet import FlaxResNet, FlaxResNetBase
from models.resnet import FlaxResNetClassifier3
from models.bridge import dsb_schedules 
from models.i2sb import DirichletFlowNetwork, ClsUnet
from collections import OrderedDict
from tqdm import tqdm
from utils import WandbLogger
from utils import model_list
from utils import batch_mul
from utils import get_config
from tqdm import tqdm
from functools import partial
import defaults_sgd
from einops import rearrange

from swag import sample_swag, sample_swag_diag
from sgd_swag import update_swag_batch_stats
from dbn_tidy import load_base_cls
from collections import namedtuple
import copy

class TrainState(train_state.TrainState):
    batch_stats: Any
    rng: Any
    ema_params: Any


def get_resnet(config, head=False):
    if config.model_name == 'FlaxResNet':
        _ResNet = partial(
            FlaxResNet if head else FlaxResNetBase,
            depth=config.model_depth,
            widen_factor=config.model_width,
            dtype=config.dtype,
            pixel_mean=defaults.PIXEL_MEAN,
            pixel_std=defaults.PIXEL_STD,
            num_classes=config.num_classes,
            num_planes=config.model_planes,
            num_blocks=tuple(
                int(b) for b in config.model_blocks.split(",")
            ) if config.model_blocks is not None else None,
            first_conv=config.first_conv,
            first_pool=config.first_pool
        )

    if config.model_style == 'BN-ReLU':
        model = _ResNet
    elif config.model_style == "FRN-Swish":
        model = partial(
            _ResNet,
            conv=partial(
                flax.linen.Conv,
                use_bias=not config.model_nobias,
                kernel_init=jax.nn.initializers.he_normal(),
                bias_init=jax.nn.initializers.zeros),
            norm=FilterResponseNorm,
            relu=flax.linen.swish)

    if head:
        return model
    return partial(model, out=config.feature_name)


def load_resnet(ckpt_dir):
    ckpt = checkpoints.restore_checkpoint(
        ckpt_dir=ckpt_dir,
        target=None
    )
    if ckpt.get("model") is not None:
        params = ckpt["model"]["params"]
        batch_stats = ckpt["model"].get("batch_stats")
        image_stats = ckpt["model"].get("image_stats")
    else:
        params = ckpt["params"]
        batch_stats = ckpt.get("batch_stats")
        image_stats = ckpt.get("image_stats")
    return ckpt, params, batch_stats, image_stats


def load_teacher(ckpt_dir):
    ckpt = checkpoints.restore_checkpoint(
        ckpt_dir=ckpt_dir,
        target=None
    )
    params = ckpt["params"]
    batch_stats = ckpt.get("batch_stats")
    config = ckpt["config"]
    return params, batch_stats, config

def load_saved(ckpt_dir):
    ckpt = checkpoints.restore_checkpoint(
        ckpt_dir=ckpt_dir,
        target=None
    )
    params = ckpt["params"]
    batch_stats = ckpt.get("batch_stats")
    config = ckpt["config"]
    tx_saved = ckpt.get("state") is not None
    return params, batch_stats, config, tx_saved


def get_classifier(config):
    feature_name = config.feature_name
    num_classes = config.num_classes

    module = FlaxResNetClassifier3

    classifier = partial(
        module,
        depth=config.model_depth,
        widen_factor=config.model_width,
        dtype=config.dtype,
        pixel_mean=defaults_sgd.PIXEL_MEAN,
        pixel_std=defaults_sgd.PIXEL_STD,
        num_classes=num_classes,
        num_planes=config.model_planes,
        num_blocks=tuple(
            int(b) for b in config.model_blocks.split(",")
        ) if config.model_blocks is not None else None,
        feature_name=feature_name,
        mimo=1,
        first_conv=config.first_conv,
        first_pool=config.first_pool
    )
    if config.model_style == "BN-ReLU":
        classifier = classifier
    elif config.model_style == "FRN-Swish":
        classifier = partial(
            classifier,
            conv=partial(
                flax.linen.Conv,
                use_bias=not config.model_nobias,
                kernel_init=jax.nn.initializers.he_normal(),
                bias_init=jax.nn.initializers.zeros
            ),
            norm=FilterResponseNorm,
            relu=flax.linen.swish
        )
    return classifier


def pdict(params, batch_stats=None, image_stats=None):
    params_dict = dict(params=params)
    if batch_stats is not None:
        params_dict["batch_stats"] = batch_stats
    if image_stats is not None:
        params_dict["image_stats"] = image_stats
    return params_dict


def get_stats(config, dataloaders, base_net, params_dict):
    feature_name = config.feature_name

    mutable = ["intermediates"]

    @partial(jax.pmap, axis_name="batch")
    def forward(batch):
        _, state = base_net.apply(
            params_dict, batch["images"], rngs=None,
            mutable=mutable,
            training=False, use_running_average=True)
        feature = state["intermediates"][feature_name][0]
        logit = state["intermediates"]["cls.logit"][0]
        logit = logit - logit.mean(axis=-1, keepdims=True)
        return feature, logit

    train_loader = dataloaders["trn_loader"](rng=None)
    train_loader = jax_utils.prefetch_to_device(train_loader, size=2)
    z_dim = None
    l_dim = None
    z_sum = 0
    l_sum = 0
    z_max = -float("inf")
    l_max = -float("inf")
    z_min = float("inf")
    l_min = float("inf")
    count = 0
    for batch in tqdm(train_loader):
        z, l = forward(batch)
        if z_dim is None:
            z_dim = z[0, 0].shape
        if l_dim is None:
            l_dim = l[0, 0].shape
        P, B, ld = l.shape
        P, B, h, w, zd = z.shape
        P, B = batch["marker"].shape
        marker = batch["marker"].reshape(-1)
        z_sum += (z.reshape(-1, h, w, zd)[marker]).sum(0).mean()
        l_sum += (l.reshape(-1, ld)[marker]).sum(0).mean()
        count += jnp.sum(marker)
        _z_max = jnp.max(z)
        _z_min = jnp.min(z)
        _l_max = jnp.max(l)
        _l_min = jnp.min(l)
        z_max = jnp.where(z_max < _z_max, _z_max, z_max)
        z_min = jnp.where(z_min > _z_min, _z_min, z_min)
        l_max = jnp.where(l_max < _l_max, _l_max, l_max)
        l_min = jnp.where(l_min > _l_min, _l_min, l_min)
    z_mean = z_sum/count
    l_mean = l_sum/count
    z_var = 0
    l_var = 0
    train_loader = dataloaders["trn_loader"](rng=None)
    train_loader = jax_utils.prefetch_to_device(train_loader, size=2)
    for batch in tqdm(train_loader):
        z, l = forward(batch)
        P, B, ld = l.shape
        P, B, h, w, zd = z.shape
        P, B = batch["marker"].shape
        marker = batch["marker"].reshape(-1)
        z_var += (
            (z.reshape(-1, h, w, zd)[marker] -
             z_mean[None, None, None, None])**2
        ).sum(0).mean()
        l_var += (
            (l.reshape(-1, ld)[marker] - l_mean[None, None])**2
        ).sum(0).mean()
    z_std = jnp.sqrt(z_var/count)
    l_std = jnp.sqrt(l_var/count)
    return (l_mean, l_std, l_max, l_min, l_dim), (z_mean, z_std, z_max, z_min, z_dim)


def get_scorenet(config):
    score_input_dim = config.score_input_dim
    in_channels = score_input_dim[-1]
    num_channels = in_channels//config.z_dim[-1] * config.n_feat

    score_func = partial(
        ClsUnet,
        num_input=config.fat,
        p_dim=config.num_classes,
        z_dim=config.z_dim,
        ch=num_channels // config.fat,
        joint=2,
        depth=config.joint_depth,
        version=config.version,
        droprate=config.droprate,
        input_scaling=config.input_scaling,
        width_multi=config.width_multi
    )
    return score_func


def get_score_input_dim(config):
    if config.fat:
        if isinstance(config.z_dim, tuple):
            h, w, d = config.z_dim
            score_input_dim = (h, w, config.fat*d)
        else:
            score_input_dim = config.fat*config.z_dim
    else:
        score_input_dim = config.z_dim

    return score_input_dim


def get_base_cls(variables, resnet_param_list):
    base_list = []
    cls_list = []
    for idx in range(len(resnet_param_list)):
        _variables = load_base_cls(
            variables, resnet_param_list, load_cls=True, base_type=idx)

        base_params = _variables["params"]["base"]
        if _variables["batch_stats"].get("base") is not None:
            base_batch_stats = _variables["batch_stats"]["base"]
        else:
            base_batch_stats = None

        cls_params = _variables["params"]["cls"]
        if _variables["batch_stats"].get("cls") is not None:
            cls_batch_stats = _variables["batch_stats"]["cls"]
        else:
            cls_batch_stats = None
        base_list.append((base_params, base_batch_stats))
        cls_list.append((cls_params, cls_batch_stats))
    return base_list, cls_list


def load_base_cls(variables, resnet_param_list, load_cls=True, base_type="A", mimo=1):
    def sorter(x):
        assert "_" in x
        name, num = x.split("_")
        return (name, int(num))

    var = variables.unfreeze()
    if not isinstance(base_type, str):
        def get(key1, key2):
            return resnet_param_list[base_type][key1][key2]
    elif base_type == "A":
        def get(key1, key2, idx=0):
            return resnet_param_list[idx][key1][key2]
    elif base_type == "AVG":
        def get(key1, key2):
            return jax.tree_util.tree_map(
                lambda *args: sum(args)/len(args),
                *[param[key1][key2] for param in resnet_param_list]
            )
    elif base_type == "BE":
        be_params = resnet_param_list[0]

        def get(key1, key2):
            return be_params[key1][key2]
    else:
        raise NotImplementedError

    resnet_params = resnet_param_list[0]
    base_param_keys = []
    res_param_keys = []
    cls_param_keys = []
    for k, v in resnet_params["params"].items():
        res_param_keys.append(k)
    for k, v in var["params"]["base"].items():
        base_param_keys.append(k)
    for k, v in var["params"]["cls"].items():
        cls_param_keys.append(k)
    res_param_keys = sorted(res_param_keys, key=sorter)
    base_param_keys = sorted(base_param_keys, key=sorter)
    cls_param_keys = sorted(cls_param_keys, key=sorter)

    cls_idx = 0
    for k in res_param_keys:
        if k in base_param_keys:
            var["params"]["base"][k] = get("params", k)
        else:
            cls_k = cls_param_keys[cls_idx]
            if load_cls:
                resnet_cls = get("params", k)
                dbn_cls = var["params"]["cls"][cls_k]
                for key, value in resnet_cls.items():
                    rank = len(value.shape)
                    if dbn_cls[key].shape[0] != value.shape[0]:
                        var["params"]["cls"][cls_k][key] = jnp.tile(
                            value, reps=[mimo]+[1]*(rank-1))
                    elif rank > 1 and dbn_cls[key].shape[1] != value.shape[1]:
                        var["params"]["cls"][cls_k][key] = jnp.tile(
                            value, reps=[1]+[mimo]+[1]*(rank-2))
                    elif rank > 2 and dbn_cls[key].shape[2] != value.shape[2]:
                        var["params"]["cls"][cls_k][key] = jnp.tile(
                            value, reps=[1, 1]+[mimo]+[1]*(rank-3))
                    elif rank > 3 and dbn_cls[key].shape[3] != value.shape[3]:
                        var["params"]["cls"][cls_k][key] = jnp.tile(
                            value, reps=[1, 1, 1]+[mimo]+[1]*(rank-4))
                    else:
                        var["params"]["cls"][cls_k][key] = value
            cls_idx += 1

    isbatchnorm = resnet_params.get("batch_stats")
    if isbatchnorm:
        base_batch_keys = []
        res_batch_keys = []
        cls_batch_keys = []
        for k, v in resnet_params["batch_stats"].items():
            res_batch_keys.append(k)
        for k, v in var["batch_stats"]["base"].items():
            base_batch_keys.append(k)
        for k, v in var["batch_stats"]["cls"].items():
            cls_batch_keys.append(k)
        res_batch_keys = sorted(res_batch_keys, key=sorter)
        base_batch_keys = sorted(base_batch_keys, key=sorter)
        cls_batch_keys = sorted(cls_batch_keys, key=sorter)

        cls_idx = 0
        for k in res_batch_keys:
            if k in base_batch_keys:
                var["batch_stats"]["base"][k] = get("batch_stats", k)
            else:
                cls_k = cls_batch_keys[cls_idx]
                if load_cls:
                    var["batch_stats"]["cls"][cls_k] = get(
                        "batch_stats", k)
                cls_idx += 1

    return freeze(var)


def build_dbn(config):
    cls_net = get_classifier(config)
    base_net = get_resnet(config, head=False)
    dsb_stats = dsb_schedules(
        config.beta1, config.beta2, config.T, linear_noise=config.linear_noise, continuous=config.dsb_continuous)
    score_net = get_scorenet(config)
    crt_net = None
    dbn = DirichletFlowNetwork(
        base_net=base_net,
        score_net=score_net,
        cls_net=cls_net,
        crt_net=crt_net,
        dsb_stats=dsb_stats,
        z_dsb_stats=None,
        fat=1,
        joint=2,
        forget=6 if config.forget!=-1 else -1,
        temp=1.,
        print_inter=False,
        mimo_cond=False,
        start_temp=config.start_temp,
        multi_mixup=False,
        continuous=False,
        centering=False,
        rf_eps=config.rf_eps,
        max_t=config.max_t,
        steps=config.T,
    )
    return dbn, (dsb_stats, None)

# @jax.jit
# def simplex_proj(seq):
#     """Algorithm from https://arxiv.org/abs/1309.1541 Weiran Wang, Miguel Á. Carreira-Perpiñán"""
#     Y = seq.reshape(-1, seq.shape[-1])
#     N, K = Y.shape
#     X = jnp.flip(jnp.sort(Y, -1), -1)
#     X_cumsum = jnp.cumsum(X, -1) - 1
#     div_seq = jnp.arange(1, K+1)
#     Xtmp = X_cumsum / div_seq

#     greater_than_Xtmp = jnp.sum((X > Xtmp), 1, keepdims=True)
#     row_indices = jnp.expand_dims(jnp.arange(N), 1)
#     selected_Xtmp = Xtmp[row_indices, greater_than_Xtmp - 1]

#     X = jnp.maximum(Y-selected_Xtmp, jnp.zeros_like(Y))
#     return jnp.reshape(X, seq.shape)

def dfm_sample(score, rng, x0, y0=None, config=None, dsb_stats=None, z_dsb_stats=None, steps=None):
    shape = x0.shape
    batch_size = shape[0]
    num_samples = config.num_samples
    n_T = config.T
    eps = config.rf_eps
    max_t = config.max_t
    timesteps = jnp.linspace(eps, max_t, n_T+1)
    # timesteps = jnp.concatenate([jnp.array([0]), timesteps], axis=0)

    rep_x0 = x0.repeat(num_samples, 0)
    rep_y0 = y0.repeat(num_samples, 0)

    @jax.jit
    def body_fn(n, x_n):
        current_t = jnp.array([timesteps[n]])
        next_t = jnp.array([timesteps[n+1]])
        current_t = jnp.tile(current_t, [batch_size*num_samples])
        next_t = jnp.tile(next_t, [batch_size*num_samples])

        eps = score(x_n, rep_y0, t=current_t)
        x_n = jax.nn.softmax(eps)
        return x_n

    x_list = [x0]
    val = rep_x0
    if steps is None:
        steps = n_T
    for i in range(0, steps):
        val = body_fn(i, val)
    x_end = val.reshape(batch_size, num_samples, -1).mean(1)
    x_list.append(x_end)

    return jnp.concatenate(x_list, axis=0)


def dsb_sample_cont(score, rng, x0, y0=None, config=None, dsb_stats=None, z_dsb_stats=None, steps=None):
    shape = x0.shape
    batch_size = shape[0]

    timesteps = jnp.concatenate(
        [jnp.linspace(1.0, 0.001, steps), jnp.zeros([1])])

    @jax.jit
    def body_fn(n, val):
        rng, x_n, x_list = val
        t, next_t = timesteps[n], timesteps[n+1]
        vec_t = jnp.ones([batch_size]) * t
        vec_next_t = jnp.ones([batch_size]) * next_t

        if config.mimo_cond:
            vec_t = jnp.tile(vec_t[:, None], reps=[1, config.fat])
        eps = score(x_n, y0, t=vec_t)

        coeffs = dsb_stats((vec_next_t, vec_t), mode='sampling')
        x_0_eps = x_n - batch_mul(coeffs['sigma_t'], eps)
        mean = batch_mul(coeffs['x0'], x_0_eps) + batch_mul(coeffs['x1'], x_n)

        rng, step_rng = jax.random.split(rng)
        h = jax.random.normal(step_rng, x_n.shape)
        x_n = mean + batch_mul(coeffs['noise'], h)

        x_list.pop(0)
        x_list.append(x_n)
        return rng, x_n, x_list

    x_list = [x0] * (steps + 1)
    # _, _, x_list = jax.lax.fori_loop(0, steps, body_fn, (rng, x0, x_list))
    val = (rng, x0, x_list)
    for i in range(0, steps):
        val = body_fn(i, val)
    _, _, x_list = val
    return jnp.concatenate(x_list, axis=0)


def launch(config, print_fn):
    # ------------------------------------------------------------------------
    # load teacher for distillation
    # ------------------------------------------------------------------------
    if config.distill:
        print(f"Loading teacher DBN {config.distill.split('/')[-1]}")
        teacher_params, teacher_batch_stats, teacher_config = load_teacher(
            config.distill)
        teacher_props = [
            "crt", "ensemble_prediction", "ensemble_exclude_a",
            "residual_prediction", "dsc", "frn_swish", "input_scaling",
            "width_multi", "dsb_continuous", "distribution",
            "beta1", "beta2", "linear_noise", "fat", "joint_depth",
            "kld_temp", "forget", "mimo_cond", "start_temp", "n_feat",
            "version", "droprate", "feature_name"
        ]
        for k in teacher_props:
            if teacher_config.get(k) is None:
                continue
            setattr(config, k, teacher_config[k])
        config.T = 1
        config.optim_lr = 0.1*teacher_config["optim_lr"]
        teacher_dsb_stats = dsb_schedules(
            teacher_config["beta1"], teacher_config["beta2"],
            teacher_config["T"], teacher_config["linear_noise"]
        )
    if config.resume:
        def props(cls):   
            return [i for i in cls.__dict__.keys() if i[:1] != '_']
        print(f"Loading checkpoint {config.resume}")
        saved_params, saved_batch_stats, saved_config, tx_saved = load_saved(
            config.resume)
        last_epoch_idx = config.last_epoch_idx
        resume = config.resume
        _config = get_config(saved_config)
        for k in props(_config):
            v = getattr(_config, k)
            setattr(config, k, v)
        config.last_epoch_idx = last_epoch_idx
        config.resume = resume

    rng = jax.random.PRNGKey(config.seed)
    model_dtype = jnp.float32
    config.dtype = model_dtype
    config.image_stats = dict(
        m=jnp.array(defaults_sgd.PIXEL_MEAN),
        s=jnp.array(defaults_sgd.PIXEL_STD))

    # ------------------------------------------------------------------------
    # load image dataset (C10, C100, TinyImageNet, ImageNet)
    # ------------------------------------------------------------------------
    dataloaders = build_dataloaders(config)
    config.num_classes = dataloaders["num_classes"]
    config.trn_steps_per_epoch = dataloaders["trn_steps_per_epoch"]

    # ------------------------------------------------------------------------
    # define and load resnet
    # ------------------------------------------------------------------------
    config.fat = 1
    swag_state_list = []
    for swag_ckpt_dir in config.swag_ckpt_dir:
        resnet_state, params, batch_stats, image_stats = load_resnet(swag_ckpt_dir)
        resnet_params = pdict(
            params=params, batch_stats=batch_stats, image_stats=image_stats)
        d = resnet_state['model']['opt_state']['1']
        swag_state = namedtuple('SWAGState', d.keys())(*d.values())
        swag_state_list.append(swag_state)

    # ------------------------------------------------------------------------
    # determine dims of base/score/cls
    # ------------------------------------------------------------------------
    resnet = get_resnet(config, head=True)
    resnet = resnet()
    _, h, w, d = dataloaders["image_shape"]
    x_dim = (h, w, d)
    config.x_dim = x_dim
    if "TinyImageNet" in config.data_name:
        zdim = (64, 64, 64)
        print(f"zdim {zdim}")
    elif "ImageNet" in config.data_name:
        zdim = (64, 64, 64)
        print(f"zdim {zdim}")
    elif "CIFAR100" in config.data_name:
        zdim = (32, 32, 64)
        print(f"zdim {zdim}")
    elif "CIFAR10" in config.data_name:
        zdim = (32, 32, 32)
        print(f"zdim {zdim}")
    else:
        print("Calculating statistics of feature z...")
        (
            (lmean, lstd, lmax, lmin, ldim),
            (zmean, zstd, zmax, zmin, zdim)
        ) = get_stats(
            config, dataloaders, resnet, resnet_params)
        print(
            f"l mean {lmean:.3f} std {lstd:.3f} max {lmax:.3f} min {lmin:.3f} dim {ldim}")
        print(
            f"z mean {zmean:.3f} std {zstd:.3f} max {zmax:.3f} min {zmin:.3f} dim {zdim}")
    config.z_dim = zdim
    score_input_dim = get_score_input_dim(config)
    config.score_input_dim = score_input_dim

    @jax.jit
    def forward_resnet(params_dict, images):
        mutable = ["intermediates"]
        _, state = resnet.apply(
            params_dict, images, rngs=None,
            mutable=mutable,
            training=False, use_running_average=True)
        features = state["intermediates"][config.feature_name][0]
        logits = state["intermediates"]["cls.logit"][0]
        return features, logits

    # ------------------------------------------------------------------------
    # define score and cls
    # ------------------------------------------------------------------------
    print("Building Diffusion Bridge Network (DBN)...")
    dbn, (dsb_stats, z_dsb_stats) = build_dbn(config)

    # ------------------------------------------------------------------------
    # initialize score & cls and replace base and cls with loaded params
    # ------------------------------------------------------------------------
    print("Initializing DBN...")
    init_rng, sub_rng = jax.random.split(rng)
    variables = dbn.init(
        {"params": init_rng, "dropout": init_rng},
        rng=init_rng,
        l_label=jnp.empty((1, config.num_classes)),
        x=jnp.empty((1, *x_dim)),
        training=False
    )
    if variables.get('batch_stats'):
        initial_batch_stats = copy.deepcopy(variables['batch_stats'])

    if config.distill:
        variables = variables.unfreeze()
        variables["params"] = teacher_params
        if variables.get("batch_stats") is not None:
            variables["batch_stats"] = teacher_batch_stats
        variables = freeze(variables)
    if config.resume:
        variables = variables.unfreeze()
        variables["params"] = saved_params
        if variables.get("batch_stats") is not None:
            variables["batch_stats"] = saved_batch_stats
        variables = freeze(variables)

    # ------------------------------------------------------------------------
    # define optimizers
    # ------------------------------------------------------------------------
    if config.decay_steps is None:
        decay_steps = config.optim_ne
    else:
        decay_steps = config.decay_steps
    scheduler = optax.warmup_cosine_decay_schedule(
        init_value=config.warmup_factor*config.optim_lr,
        peak_value=config.optim_lr,
        warmup_steps=config.warmup_steps,
        decay_steps=decay_steps * config.trn_steps_per_epoch)

    if config.optim_base == "adam":
        base_optim = partial(
            optax.adamw, learning_rate=scheduler, weight_decay=config.optim_weight_decay
        )
    elif config.optim_base == "sgd":
        base_optim = partial(optax.sgd, learning_rate=scheduler,
                             momentum=config.optim_momentum)
    else:
        raise NotImplementedError
    partition_optimizers = {
        "base": optax.set_to_zero() if not config.train_base else base_optim(),
        "score": base_optim(),
        "cls": optax.set_to_zero(),
        "crt": base_optim()
    }

    def tagging(path, v):
        def cls_list():
            for i in range(config.fat):
                if f"cls_{i}" in path:
                    return True
            return False
        if "base" in path:
            return "base"
        elif "score" in path:
            return "score"
        elif "cls" in path:
            return "cls"
        elif "crt" in path:
            return "crt"
        elif cls_list():
            return "cls"
        else:
            raise NotImplementedError
    partitions = flax.core.freeze(
        flax.traverse_util.path_aware_map(tagging, variables["params"]))
    optimizer = optax.multi_transform(partition_optimizers, partitions)

    # ------------------------------------------------------------------------
    # create train state
    # ------------------------------------------------------------------------
    state_rng, sub_rng = jax.random.split(sub_rng)
    state = TrainState.create(
        apply_fn=dbn.apply,
        params=variables["params"],
        ema_params=variables["params"],
        tx=optimizer,
        batch_stats=variables.get("batch_stats"),
        rng=state_rng
    )
    if config.resume:
        last_epochs = config.last_epoch_idx + 1
        saved_iteration = config.trn_steps_per_epoch * last_epochs
        state = state.replace(step=saved_iteration)
        if tx_saved:
            print("Load saved optimizer")
            _ckpt = dict(
                params=state.params,
                batch_stats=state.batch_stats,
                config=dict(),
                state=state
            )
            ckpt = checkpoints.restore_checkpoint(
                ckpt_dir=config.resume,
                target=_ckpt
            )
            state = state.replace(tx=ckpt["state"].tx)
            del _ckpt
            del ckpt


    # ------------------------------------------------------------------------
    # define sampler and metrics
    # ------------------------------------------------------------------------
    @jax.jit
    def mse_loss(noise, output):
        p = config.mse_power
        sum_axis = list(range(1, len(output.shape[1:])+1))
        loss = jnp.sum(jnp.abs(noise-output)**p, axis=sum_axis)
        return loss

    @jax.jit
    def pseudohuber_loss(noise, output):
        sum_axis = list(range(1, len(output.shape[1:])+1))
        loss = jnp.sum(jnp.sqrt((noise-output)**2 + 0.003**2) - 0.003, axis=sum_axis)
        return loss
    
    @jax.jit
    def ce_loss(logits, labels):
        target = common_utils.onehot(labels, num_classes=logits.shape[-1])
        pred = jax.nn.log_softmax(logits, axis=-1)
        loss = -jnp.sum(target*pred, axis=-1)
        return loss

    @jax.jit
    def kld_loss(target_list, refer_list, T=1.):
        if not isinstance(target_list, list):
            target_list = [target_list]
        if not isinstance(refer_list, list):
            refer_list = [refer_list]
        kld_sum = 0
        count = 0
        for tar, ref in zip(target_list, refer_list):
            logq = jax.nn.log_softmax(tar/T, axis=-1)
            logp = jax.nn.log_softmax(ref/T, axis=-1)
            q = jnp.exp(logq)
            integrand = q*(logq-logp)
            kld = jnp.sum(integrand, axis=-1)
            kld_sum += T**2*kld
            count += 1
        return kld_sum/count

    @jax.jit
    def self_kld_loss(target_list, T=1.):
        N = len(target_list)
        kld_sum = 0
        count = N*(N-1)
        logprob_list = [jax.nn.log_softmax(
            tar/T, axis=-1) for tar in target_list]
        for logq in logprob_list:
            for logp in logprob_list:
                q = jnp.exp(logq)
                integrand = q*(logq-logp)
                kld = jnp.sum(integrand, axis=-1)
                kld_sum += T**2*kld
        return kld_sum/count

    @jax.jit
    def reduce_mean(loss, marker):
        count = jnp.sum(marker)
        loss = jnp.where(marker, loss, 0).sum()
        loss = jnp.where(count != 0, loss/count, loss)
        return loss

    @jax.jit
    def reduce_sum(loss, marker):
        loss = jnp.where(marker, loss, 0).sum()
        return loss

    def kld_loss_fn(t, r, marker):
        return reduce_sum(
            kld_loss(t, r), marker)

    def self_kld_loss_fn(t, marker):
        return reduce_sum(self_kld_loss(t), marker)

    # ------------------------------------------------------------------------
    # define step collecting features and logits
    # ------------------------------------------------------------------------
    @partial(jax.pmap, axis_name="batch")
    def step_label(state, batch):
        rng = state.rng
        swag_param_list = []
        for swag_state in swag_state_list:
            swag_params_per_seed = sample_swag_diag(config.num_target_samples, rng, swag_state)
            swag_param_list += swag_params_per_seed
            rng, _ = jax.random.split(rng)
            
        z0_list = []
        logitsA = []
        for swag_param in swag_param_list:
            res_params_dict = dict(params=swag_param)
            if image_stats is not None:
                res_params_dict["image_stats"] = image_stats
            if batch_stats is not None:
                # Update batch_stats
                trn_loader = dataloaders['dataloader'](rng=state.rng)
                trn_loader = jax_utils.prefetch_to_device(trn_loader, size=2)
                iter_batch_stats = initial_batch_stats
                for _, batch in enumerate(trn_loader, start=1):
                    iter_batch_stats = update_swag_batch_stats(resnet, res_params_dict, batch, iter_batch_stats)
                res_params_dict["batch_stats"] = cross_replica_mean(iter_batch_stats)
            
            images = batch["images"]
            z0, logits0 = forward_resnet(res_params_dict, images)
            z0_list.append(z0)
            # logits0: (B, d)
            logits0 = logits0 - logits0.mean(-1, keepdims=True)
            logitsA.append(logits0)
        logitsB = logitsA[0]
        zB = z0_list[0]
        zA = jnp.concatenate(z0_list, axis=-1)

        _logitsA = jnp.stack(logitsA)
        logprobs = jax.nn.log_softmax(_logitsA, axis=-1)
        ens_logprobs = jax.scipy.special.logsumexp(
            logprobs, axis=0) - np.log(logprobs.shape[0])
        ens_logits = ens_logprobs - ens_logprobs.mean(-1, keepdims=True)
        logitsA = [ens_logits]

        batch["zB"] = zB
        batch["zA"] = zA
        batch["logitsB"] = logitsB
        batch["logitsA"] = logitsA
        if config.distill:
            drop_rng, score_rng = jax.random.split(state.rng)
            params_dict = pdict(
                params=teacher_params,
                image_stats=config.image_stats,
                batch_stats=teacher_batch_stats
            )
            rngs_dict = dict(dropout=drop_rng)
            model_bd = dbn.bind(params_dict, rngs=rngs_dict)
            teacher_steps = teacher_config["T"]
            _dfm_sample = partial(
                dfm_sample,
                config=EasyDict(teacher_config),
                dsb_stats=teacher_dsb_stats,
                z_dsb_stats=None,
                steps=teacher_steps)
            logitsC, _ = model_bd.sample(
                score_rng, _dfm_sample, batch["images"])
            logitsC = rearrange(
                logitsC, "n (t b) z -> t b (n z)", t=teacher_steps+1)
            batch["logitsC"] = logitsC[-1]
        return batch

    # ------------------------------------------------------------------------
    # define train step
    # ------------------------------------------------------------------------

    def loss_func(params, state, batch, train=True):
        logitsA = batch["logitsA"]
        logitsA = [l for l in batch["logitsA"]]  # length fat list
        _logitsA = jnp.concatenate(
            logitsA, axis=-1)  # (B, fat*num_classes)
        drop_rng, score_rng = jax.random.split(state.rng)
        params_dict = pdict(
            params=params,
            image_stats=config.image_stats,
            batch_stats=state.batch_stats)
        rngs_dict = dict(dropout=drop_rng)
        mutable = ["batch_stats"]
        
        output = state.apply_fn(
            params_dict, score_rng,
            _logitsA, batch["images"],
            training=train,
            rngs=rngs_dict,
            **(dict(mutable=mutable) if train else dict()),
        )
        new_model_state = output[1] if train else None
        eps, diff  = output[0] if train else output

        score_loss = mse_loss(eps, diff)
        # score_loss = pseudohuber_loss(epsilon, diff)
        if batch.get("logitsC") is not None:
            logitsC = batch["logitsC"]
            a = config.distill_alpha
            score_loss = (
                a*mse_loss(eps, (l1-logitsC)) + (1-a)*score_loss
            )
        score_loss = reduce_mean(score_loss, batch["marker"])
        total_loss = config.gamma*score_loss

        count = jnp.sum(batch["marker"])
        metrics = OrderedDict({
            "loss": total_loss*count,
            "score_loss": score_loss*count,
            "count": count,
        })

        return total_loss, (metrics, new_model_state)

    def get_loss_func():
        return loss_func

    @ partial(jax.pmap, axis_name="batch")
    def step_train(state, batch):

        def loss_fn(params):
            _loss_func = get_loss_func()
            return _loss_func(params, state, batch)

        # swag_param = sample_swag(1, state.rng, swag_state)[0]
        # res_params_dict = dict(params=swag_param)
        res_params_dict = dict(params=swag_state_list[0].mean)
        if image_stats is not None:
            res_params_dict["image_stats"] = image_stats
        if batch_stats is not None:
            # Update batch_stats
            trn_loader = dataloaders['dataloader'](rng=state.rng)
            trn_loader = jax_utils.prefetch_to_device(trn_loader, size=2)
            iter_batch_stats = initial_batch_stats
            for _, batch in enumerate(trn_loader, start=1):
                iter_batch_stats = update_swag_batch_stats(resnet, res_params_dict, batch, iter_batch_stats)
            res_params_dict["batch_stats"] = cross_replica_mean(iter_batch_stats)
        
        new_variables = load_base_cls(variables, [res_params_dict], load_cls=True, base_type="A", mimo=1)
        state = state.replace(params=freeze(dict(base=new_variables["params"]["base"], 
                                              cls=new_variables["params"]["cls"], 
                                              score=state.params["score"])),)
        assert new_variables["batch_stats"].get("base") is None
        assert new_variables["batch_stats"].get("cls") is None

        (loss, (metrics, new_model_state)), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(state.params)
        grads = jax.lax.pmean(grads, axis_name="batch")

        new_state = state.apply_gradients(
            grads=grads, batch_stats=new_model_state.get("batch_stats"))
        a = config.ema_decay
        def update_ema(wt, ema_tm1): return jnp.where(
            (wt != ema_tm1) & (a < 1), a*wt + (1-a)*ema_tm1, wt)
        new_state = new_state.replace(
            ema_params=jax.tree_util.tree_map(
                update_ema,
                new_state.params,
                new_state.ema_params))
        metrics = jax.lax.psum(metrics, axis_name="batch")
        return new_state, metrics

    # ------------------------------------------------------------------------
    # define valid step
    # ------------------------------------------------------------------------
    @ partial(jax.pmap, axis_name="batch")
    def step_valid(state, batch):

        _loss_func = get_loss_func()
        _, (metrics, _) = _loss_func(
            state.ema_params, state, batch, train=False)

        metrics = jax.lax.psum(metrics, axis_name="batch")
        return metrics

    # ------------------------------------------------------------------------
    # define sampling step
    # ------------------------------------------------------------------------
    def ensemble_accnll(logits, labels, marker):
        acc_list = []
        nll_list = []
        cum_acc_list = []
        cum_nll_list = []
        prob_sum = 0
        for i, prob in enumerate(logits):
            prob_sum += prob
            if i != 0:  # Not B
                acc = evaluate_acc(
                    prob, labels, log_input=False, reduction="none")
                nll = evaluate_nll(
                    prob, labels, log_input=False, reduction='none')
                acc = reduce_sum(acc, marker)
                nll = reduce_sum(nll, marker)
                acc_list.append(acc)
                nll_list.append(nll)
                avg_prob = prob_sum / (i+1)
                acc = evaluate_acc(
                    avg_prob, labels, log_input=False, reduction="none")
                nll = evaluate_nll(
                    avg_prob, labels, log_input=False, reduction='none')
                acc = reduce_sum(acc, marker)
                nll = reduce_sum(nll, marker)
                cum_acc_list.append(acc)
                cum_nll_list.append(nll)
        return acc_list, nll_list, cum_acc_list, cum_nll_list

    def sample_func(state, batch, steps=(config.T+1)//2):
        # swag_param = sample_swag(1, state.rng, swag_state)[0]
        # res_params_dict = dict(params=swag_param)
        res_params_dict = dict(params=swag_state_list[0].mean)
        if image_stats is not None:
            res_params_dict["image_stats"] = image_stats
        if batch_stats is not None:
            # Update batch_stats
            trn_loader = dataloaders['dataloader'](rng=state.rng)
            trn_loader = jax_utils.prefetch_to_device(trn_loader, size=2)
            iter_batch_stats = initial_batch_stats
            for _, batch in enumerate(trn_loader, start=1):
                iter_batch_stats = update_swag_batch_stats(resnet, res_params_dict, batch, iter_batch_stats)
            res_params_dict["batch_stats"] = cross_replica_mean(iter_batch_stats)

        # splitted_swag_param = split_base_cls(variables, [res_params_dict],
        #                              *key_list, load_cls=True, base_type="A", mimo=1)
        new_variables = load_base_cls(variables, [res_params_dict], load_cls=True, base_type="A", mimo=1)
        state = state.replace(ema_params=dict(base=new_variables["params"]["base"], 
                                              cls=new_variables["params"]["cls"], 
                                              score=state.ema_params["score"]),)
        assert new_variables["batch_stats"].get("base") is None
        assert new_variables["batch_stats"].get("cls") is None
        
        labels = batch["labels"]
        drop_rng, score_rng = jax.random.split(state.rng)

        params_dict = pdict(
            params=state.ema_params,
            image_stats=config.image_stats,
            batch_stats=state.batch_stats)
        rngs_dict = dict(dropout=drop_rng)
        model_bd = dbn.bind(params_dict, rngs=rngs_dict)
        _dfm_sample = partial(
            dfm_sample,
            config=config, dsb_stats=dsb_stats, z_dsb_stats=z_dsb_stats, steps=steps)
        logitsC, _zC = model_bd.sample(
            score_rng, _dfm_sample, batch["images"])
        logitsC = rearrange(logitsC, "n (t b) z -> t n b z", t=2)

        (
            acc_list, nll_list, cum_acc_list, cum_nll_list
        ) = ensemble_accnll([logitsC[i][0] for i in range(logitsC.shape[0])], labels, batch["marker"])
        metrics = OrderedDict({
            "count": jnp.sum(batch["marker"]),
        })

        for i, (acc, nll, ensacc, ensnll) in enumerate(
                zip(acc_list, nll_list, cum_acc_list, cum_nll_list), start=1):
            metrics[f"acc{i}"] = acc
            metrics[f"nll{i}"] = nll
            metrics[f"ens_acc{i}"] = ensacc
            metrics[f"ens_nll{i}"] = ensnll
        return metrics

    @ partial(jax.pmap, axis_name="batch")
    def step_sample(state, batch):
        steps = config.T
        metrics = sample_func(state, batch, steps=steps)
        metrics = jax.lax.psum(metrics, axis_name="batch")
        return metrics

    @partial(jax.pmap, axis_name="batch")
    def step_sample_valid(state, batch):
        steps = config.T
        metrics = sample_func(state, batch, steps=steps)
        metrics = jax.lax.psum(metrics, axis_name="batch")
        return metrics

    @ jax.jit
    def accnll(logits, labels, marker):
        logprob = jax.nn.log_softmax(logits, axis=-1)
        acc = evaluate_acc(
            logprob, labels, log_input=True, reduction="none")
        nll = evaluate_nll(
            logprob, labels, log_input=True, reduction='none')
        acc = reduce_sum(acc, marker)
        nll = reduce_sum(nll, marker)
        return acc, nll

    # ------------------------------------------------------------------------
    # define mixup
    # ------------------------------------------------------------------------
    @partial(jax.pmap, axis_name="batch")
    def step_mixup(state, batch):
        count = jnp.sum(batch["marker"])
        x = batch["images"]
        y = batch["labels"]
        batch_size = x.shape[0]
        a = config.mixup_alpha
        beta_rng, perm_rng = jax.random.split(state.rng)

        lamda = jnp.where(a > 0, jax.random.beta(beta_rng, a, a), 1)

        perm_x = jax.random.permutation(perm_rng, x)
        perm_y = jax.random.permutation(perm_rng, y)
        mixed_x = (1-lamda)*x+lamda*perm_x
        mixed_y = jnp.where(lamda < 0.5, y, perm_y)
        mixed_x = jnp.where(count == batch_size, mixed_x, x)
        mixed_y = jnp.where(count == batch_size, mixed_y, y)

        batch["images"] = mixed_x
        batch["labels"] = mixed_y
        return batch

    def choose_what_to_print(print_bin, inter):
        prev_score, prev_arr = print_bin
        new_score = ((inter[-1]-inter[-2])**2).sum(1) - \
            ((inter[0]-inter[-1])**2).sum(1)
        for i in range(config.fat-1):
            new_score += -(
                (inter[-1, :, i*10:(i+1)*10] -
                 inter[-1, :, (i+1)*10:(i+2)*10])**2
            ).sum(1)
        new_score_idx = jnp.argmin(new_score)
        if jnp.any(prev_score > new_score[new_score_idx]):
            new_arr = inter[:, new_score_idx, :]
            new_score = new_score[new_score_idx]
        else:
            new_score = prev_score
            new_arr = prev_arr
        return new_score, new_arr
    # ------------------------------------------------------------------------
    # init settings and wandb
    # ------------------------------------------------------------------------
    cross_replica_mean = jax.pmap(lambda x: jax.lax.pmean(x, 'x'), 'x')
    best_acc = -float("inf")
    train_summary = dict()
    valid_summary = dict()
    test_summary = dict()
    state = jax_utils.replicate(state)

    wandb.init(
        project="dbn",
        config=vars(config),
        mode="disabled" if config.nowandb else "online"
    )
    wandb.define_metric("val/loss", summary="min")
    wandb.define_metric("val/acc", summary="max")
    wandb.define_metric("val/ens_acc", summary="max")
    wandb.define_metric("val/ens_nll", summary="min")
    wandb.define_metric(f"val/ens_acc{config.fat}", summary="max")
    wandb.define_metric(f"val/ens_nll{config.fat}", summary="min")
    wandb.run.summary["params_base"] = (
        sum(x.size for x in jax.tree_util.tree_leaves(
            variables["params"]["base"]))
    )
    wandb.run.summary["params_score"] = (
        sum(x.size for x in jax.tree_util.tree_leaves(
            variables["params"]["score"]))
    )
    wandb.run.summary["params_cls"] = (
        sum(x.size for x in jax.tree_util.tree_leaves(
            variables["params"]["cls"]))
    )
    # params_flatten = flax.traverse_util.flatten_dict(
    #     variables["params"]["score"])
    # for k, v in params_flatten.items():
    #     print("score", k, v.shape, flush=True)
    wl = WandbLogger()

    def summarize_metrics(metrics, key="trn"):
        metrics = common_utils.get_metrics(metrics)
        summarized = {
            f"{key}/{k}": v for k, v in jax.tree_util.tree_map(lambda e: e.sum(0), metrics).items()}
        inter_samples = False
        for k, v in summarized.items():
            if "count" in k:
                continue
            elif "lr" in k:
                continue
            if "_inter" in k:
                inter_samples = True
            summarized[k] /= summarized[f"{key}/count"]
        if inter_samples and key == "tst":
            assert summarized.get(f"tst/acc{config.T+1}_inter") is None
            acc_str = ",".join(
                [f"ACC({config.T+1-i})   {summarized[f'{key}/acc{i}_inter']:.4f}" for i in range(1, config.T+1)])
            ens_acc_str = ",".join(
                [f"ENS({config.T+1}to{config.T+1-i}){summarized[f'{key}/ens_acc{i}_inter']:.4f}" for i in range(1, config.T+1)])
            print(acc_str, flush=True)
            print(ens_acc_str, flush=True)
        if inter_samples:
            for i in range(1, config.T+1):
                del summarized[f"{key}/acc{i}_inter"]
                del summarized[f"{key}/ens_acc{i}_inter"]

        del summarized[f"{key}/count"]
        return summarized

    for epoch_idx in tqdm(range(config.last_epoch_idx, config.optim_ne)):
        epoch_rng = jax.random.fold_in(sub_rng, epoch_idx)
        train_rng, valid_rng, test_rng = jax.random.split(epoch_rng, 3)

        valid_only_cond =(config.last_epoch_idx > 0 and epoch_idx==config.last_epoch_idx) 
        # ------------------------------------------------------------------------
        # train by getting features from each resnet
        # ------------------------------------------------------------------------
        if not valid_only_cond:
            train_loader = dataloaders["dataloader"](rng=epoch_rng)
            train_loader = jax_utils.prefetch_to_device(train_loader, size=2)
            train_metrics = []
            print_bin = (float("inf"), None)
            for batch_idx, batch in enumerate(train_loader):
                batch_rng = jax.random.fold_in(train_rng, batch_idx)
                state = state.replace(rng=jax_utils.replicate(batch_rng))
                if config.mixup_alpha > 0:
                    batch = step_mixup(state, batch)
                batch = step_label(state, batch)
                state, metrics = step_train(state, batch)
                train_metrics.append(metrics)

            train_summarized = summarize_metrics(train_metrics, "trn")
            train_summary.update(train_summarized)
            train_summary["trn/lr"] = scheduler(jax_utils.unreplicate(state.step))
            wl.log(train_summary)

            if state.batch_stats is not None:
                state = state.replace(
                    batch_stats=cross_replica_mean(state.batch_stats))

        # ------------------------------------------------------------------------
        # valid by getting features from each resnet
        # ------------------------------------------------------------------------
        valid_loader = dataloaders["val_loader"](rng=None)
        valid_loader = jax_utils.prefetch_to_device(valid_loader, size=2)
        valid_metrics = []
        for batch_idx, batch in enumerate(valid_loader):
            batch_rng = jax.random.fold_in(valid_rng, batch_idx)
            state = state.replace(rng=jax_utils.replicate(batch_rng))
            # batch = step_label(state, batch)
            # metrics = step_valid(state, batch)
            acc_metrics = step_sample_valid(state, batch)
            metrics = acc_metrics
            valid_metrics.append(metrics)

        valid_summarized = summarize_metrics(valid_metrics, "val")
        valid_summary.update(valid_summarized)
        wl.log(valid_summary)

        # ------------------------------------------------------------------------
        # test by getting features from each resnet
        # ------------------------------------------------------------------------
        criteria = valid_summarized[f"val/acc1"]
        if best_acc < criteria:
            if valid_only_cond:
                best_acc = criteria
                continue
            test_loader = dataloaders["tst_loader"](rng=None)
            test_loader = jax_utils.prefetch_to_device(test_loader, size=2)
            test_metrics = []
            for batch_idx, batch in enumerate(test_loader):
                batch_rng = jax.random.fold_in(test_rng, batch_idx)
                state = state.replace(rng=jax_utils.replicate(batch_rng))
                # batch = step_label(state, batch)
                # metrics = step_valid(state, batch)
                acc_metrics = step_sample(state, batch)
                metrics = acc_metrics
                test_metrics.append(metrics)

            test_summarized = summarize_metrics(test_metrics, "tst")
            test_summary.update(test_summarized)
            wl.log(test_summary)
            best_acc = criteria
            if config.save:
                save_path = config.save
                save_state = jax_utils.unreplicate(state)
                if getattr(config, "dtype", None) is not None:
                    config.dtype = str(config.dtype)
                ckpt = dict(
                    params=save_state.ema_params,
                    batch_stats=save_state.batch_stats,
                    config=vars(config),
                    state=save_state
                )
                orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
                checkpoints.save_checkpoint(
                    ckpt_dir=save_path,
                    target=ckpt,
                    step=epoch_idx,
                    overwrite=True,
                    orbax_checkpointer=orbax_checkpointer
                )

        wl.flush()

        jax.random.normal(rng, ()).block_until_ready()
        if jnp.isnan(train_summarized["trn/loss"]):
            print("NaN detected")
            break

    wandb.finish()


def main():

    TIME_STAMP = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    parser = defaults.default_argument_parser()

    parser.add_argument("--config", default=None, type=str)
    args, argv = parser.parse_known_args(sys.argv[1:])
    if args.config is not None:
        import yaml
        with open(args.config, 'r') as f:
            arg_defaults = yaml.safe_load(f)

    parser.add_argument("--model_planes", default=16, type=int)
    parser.add_argument("--model_blocks", default=None, type=str)
    parser.add_argument("--model_nobias", action="store_true")
    parser.add_argument('--first_conv', nargs='+', type=int)
    parser.add_argument('--first_pool', nargs='+', type=int)
    # ---------------------------------------------------------------------------------------
    # optimizer
    # ---------------------------------------------------------------------------------------
    parser.add_argument('--optim_ne', default=350, type=int,
                        help='the number of training epochs (default: 200)')
    parser.add_argument('--optim_lr', default=2e-4, type=float,
                        help='base learning rate (default: 1e-4)')
    parser.add_argument('--warmup_factor', default=0.01, type=float)
    parser.add_argument('--warmup_steps', default=0, type=int)
    parser.add_argument('--decay_steps', default=None, type=int)
    parser.add_argument('--optim_momentum', default=0.9, type=float,
                        help='momentum coefficient (default: 0.9)')
    parser.add_argument('--optim_weight_decay', default=0.1, type=float,
                        help='weight decay coefficient (default: 0.0001)')
    parser.add_argument('--optim_base', default="adam", type=str)
    parser.add_argument('--start_cls', default=-1, type=int)
    parser.add_argument('--cls_optim', default="adam", type=str)
    parser.add_argument('--start_base', default=999999999999, type=int)
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--last_epoch_idx', default=0, type=int)
    # ---------------------------------------------------------------------------------------
    # training
    # ---------------------------------------------------------------------------------------
    parser.add_argument('--save', default=None, type=str)
    parser.add_argument('--seed', default=2023, type=int)
    parser.add_argument("--tag", default=None, type=str)
    parser.add_argument("--gamma", default=1, type=float)
    parser.add_argument("--lgamma", default=1, type=float)
    parser.add_argument("--zgamma", default=1, type=float)
    parser.add_argument("--beta", default=1, type=float)
    parser.add_argument("--ema_decay", default=0.9999, type=float)
    parser.add_argument("--mse_power", default=2, type=int)
    parser.add_argument("--crt", default=0, type=int)
    parser.add_argument("--cls_from_scratch", action="store_true")
    parser.add_argument("--mixup_alpha", default=0, type=float)
    # ---------------------------------------------------------------------------------------
    # experiemnts
    # ---------------------------------------------------------------------------------------
    parser.add_argument("--nowandb", action="store_true")
    parser.add_argument("--train_base", action="store_true")
    parser.add_argument("--base_type", default="A", type=str)
    parser.add_argument("--best_nll", action="store_true")
    # replace base with LatentBE-trained model (directory of checkpoint)
    parser.add_argument("--latentbe", default=None, type=str)
    # MIMO classifier
    parser.add_argument("--mimo_cls", default=1, type=int)
    # Ensemble Prediction
    parser.add_argument("--ensemble_prediction", default=0, type=int)
    parser.add_argument("--ensemble_exclude_a", action="store_true")
    # Residual (logitB-logitA) Prediction
    parser.add_argument("--residual_prediction", action="store_true")
    # Depthwise Seperable Conv
    parser.add_argument("--dsc", action="store_true")
    # get intermediate samples
    parser.add_argument("--medium", action="store_true")
    # FiterResponseNorm and Swish activation in Score net
    parser.add_argument("--frn_swish", action="store_true")
    # Score net input scaling (equivalent to NN normalization)
    parser.add_argument("--input_scaling", default=1, type=float)
    # Score net width multiplier
    parser.add_argument("--width_multi", default=1, type=float)
    # print intermediate samples
    parser.add_argument("--print_inter", action="store_true")
    # continuous diffusion
    parser.add_argument("--dsb_continuous", action="store_true")
    # distribution type; 0: 1to1, 1: mixup, 2: mode, 3: annealing
    parser.add_argument("--distribution", default=0, type=int)
    # logit centering for logitsB
    parser.add_argument("--centering", action="store_true")
    # ---------------------------------------------------------------------------------------
    # diffusion
    # ---------------------------------------------------------------------------------------
    parser.add_argument("--T", default=5, type=int)
    parser.add_argument("--beta1", default=1e-4, type=float)
    parser.add_argument("--beta2", default=3e-4, type=float)
    parser.add_argument("--linear_noise", action="store_true")
    # Fat DSB (A+A -> B1+B2)
    parser.add_argument("--fat", default=1, type=int)
    # joint diffusion
    parser.add_argument("--joint_depth", default=1, type=int)
    parser.add_argument("--kld_joint", action="store_true")
    parser.add_argument("--kld_temp", default=1, type=float)
    parser.add_argument("--forget", default=0, type=int)
    parser.add_argument("--prob_loss", action="store_true")
    parser.add_argument("--ce_loss", default=0, type=float)
    parser.add_argument("--mimo_cond", action="store_true")
    parser.add_argument("--start_temp", default=1, type=float)
    # Progressive Distillation
    parser.add_argument("--distill", default=None, type=str)
    parser.add_argument("--distill_alpha", default=1, type=float)
    # Gaussian prior (rebuttal)
    parser.add_argument("--gaussian_prior", action="store_true")
    # Rectified Flow (rebuttal)
    parser.add_argument("--rf_eps", default=1e-3, type=float)
    # ---------------------------------------------------------------------------------------
    # networks
    # ---------------------------------------------------------------------------------------
    parser.add_argument("--n_feat", default=256, type=int)
    parser.add_argument("--version", default="v1.0", type=str)
    parser.add_argument("--droprate", default=0, type=float)
    parser.add_argument("--feature_name", default="feature.layer3stride2")
    # UNetModel only
    parser.add_argument("--learn_sigma", action="store_true")
    parser.add_argument("--num_res_blocks", default=1, type=int)
    parser.add_argument("--num_heads", default=1, type=int)
    parser.add_argument("--use_scale_shift_norm", action="store_true")
    parser.add_argument("--resblock_updown", action="store_true")
    parser.add_argument("--use_new_attention_order", action="store_true")
    parser.add_argument("--large", action="store_true")

    if args.config is not None:
        parser.set_defaults(**arg_defaults)

    args = parser.parse_args()

    if args.seed < 0:
        args.seed = (
            os.getpid()
            + int(datetime.datetime.now().strftime('%S%f'))
            + int.from_bytes(os.urandom(2), 'big')
        )

    # if args.save is not None:
    #     args.save = os.path.abspath(args.save)
    #     if os.path.exists(args.save):
    #         raise AssertionError(f'already existing args.save = {args.save}')
    #     os.makedirs(args.save, exist_ok=True)

    print_fn = partial(print, flush=True)
    # if args.save:
    #     def print_fn(s):
    #         with open(os.path.join(args.save, f'{TIME_STAMP}.log'), 'a') as fp:
    #             fp.write(s + '\n')
    #         print(s, flush=True)

    log_str = tabulate([
        ('sys.platform', sys.platform),
        ('Python', sys.version.replace('\n', '')),
        ('JAX', jax.__version__ + ' @' + os.path.dirname(jax.__file__)),
        ('jaxlib', jaxlib.__version__ + ' @' + os.path.dirname(jaxlib.__file__)),
        ('Flax', flax.__version__ + ' @' + os.path.dirname(flax.__file__)),
        ('Optax', optax.__version__ + ' @' + os.path.dirname(optax.__file__)),
    ]) + '\n'
    log_str = f'Environments:\n{log_str}'
    log_str = datetime.datetime.now().strftime(
        '[%Y-%m-%d %H:%M:%S] ') + log_str
    print_fn(log_str)

    log_str = ''
    max_k_len = max(map(len, vars(args).keys()))
    for k, v in vars(args).items():
        log_str += f'- args.{k.ljust(max_k_len)} : {v}\n'
    log_str = f'Command line arguments:\n{log_str}'
    log_str = datetime.datetime.now().strftime(
        '[%Y-%m-%d %H:%M:%S] ') + log_str
    print_fn(log_str)

    if jax.local_device_count() > 1:
        log_str = f'Multiple local devices are detected:\n{jax.local_devices()}\n'
        log_str = datetime.datetime.now().strftime(
            '[%Y-%m-%d %H:%M:%S] ') + log_str
        print_fn(log_str)

    launch(args, print_fn)


if __name__ == '__main__':
    # import traceback
    # with open("error.log", "w") as log:
    #     try:
    #         main()
    #     except Exception:
    #         print("ERROR OCCURED! Check it out in error.log.")
    #         traceback.print_exc(file=log)
    main()
