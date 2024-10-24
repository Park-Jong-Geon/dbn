from re import M
import time
import os
import math
import orbax
from cls_plot import cls_plot


from typing import Any, Tuple

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
from data.build import build_dataloaders, _build_dataloader, _build_featureloader
from giung2.metrics import evaluate_acc, evaluate_nll
from utils import evaluate_top2acc, evaluate_topNacc
from models.resnet import FlaxResNetClassifier, FlaxResNetClassifier2
from models.bridge import CorrectionModel, FeatureUnet, dsb_schedules, MLP
from models.i2sb import UNetModel
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm
from utils import WandbLogger, pixelize, normalize_logits, unnormalize_logits
from utils import model_list, logit_dir_list, feature_dir_list, feature2_dir_list
from utils import get_info_in_dir, jprint, expand_to_broadcast, FeatureBank
from tqdm import tqdm
from functools import partial

from utils import batch_mul, batch_add


def build_featureloaders(config, rng=None):
    def load_arrays(dir, div, prop, mode_idx, i, length=None):
        path = f"{dir}/{div}_{prop}_M{mode_idx}S{i}.npy"
        if not os.path.exists(path):
            return jnp.ones((length,))
        with open(path, "rb") as f:
            logits = np.load(f)
        return logits

    def get_context_dir(dir):
        f_dir = dir.split("_")[0]
        f_dir = dir.replace(f_dir, f"{f_dir}_last")
        return f_dir

    def get_ens_logits(logits, mean_axis=0):
        logitmean = jnp.mean(logits[mean_axis], axis=-1)[:, None]
        ens_prob = 0
        for l in logits:
            ens_prob += jax.nn.softmax(l, axis=-1)
        ens_prob /= len(logits)
        ens_logprob = jnp.log(ens_prob)
        ens_logprob_mean = jnp.mean(ens_logprob, axis=-1)[:, None]
        ens_logits = ens_logprob-ens_logprob_mean+logitmean
        return ens_logits

    def get_avg_logits(logits):
        return sum(logits)/len(logits)

    dir = config.features_dir
    f_dir = get_context_dir(dir)
    if config.context:
        # assert f_dir == config.contexts_dir
        f_dir = config.contexts_dir
    # B: current mode, A: other modes
    n_Amodes = config.n_Amodes
    n_samples_each_mode = config.n_samples_each_mode
    n_samples_each_Bmode = n_Amodes*n_samples_each_mode
    n_samples_each_Amode = n_samples_each_mode
    train_Alogits_list = []
    train_Blogits_list = []
    train_Afeatures_list = []
    train_Bfeatures_list = []
    train_Asupples_list = []
    train_Bsupples_list = []
    train_lambdas_list = []
    train_lambdas_checker = []
    train_lambdas_checker2 = []
    valid_Alogits_list = []
    valid_Blogits_list = []
    valid_Afeatures_list = []
    valid_Bfeatures_list = []
    valid_Asupples_list = []
    valid_Bsupples_list = []
    valid_lambdas_list = []
    test_Alogits_list = []
    test_Blogits_list = []
    test_Afeatures_list = []
    test_Bfeatures_list = []
    test_Asupples_list = []
    test_Bsupples_list = []
    test_lambdas_list = []

    def putinB(mode_idx, i):
        # logits
        train_logits = load_arrays(
            dir, "train", "features", mode_idx, i)
        valid_logits = load_arrays(
            dir, "valid", "features", mode_idx, i)
        test_logits = load_arrays(
            dir, "test", "features", mode_idx, i)
        train_Blogits_list.append(train_logits)
        valid_Blogits_list.append(valid_logits)
        test_Blogits_list.append(test_logits)
        # lambdas
        train_lambdas = load_arrays(
            dir, "train", "lambdas", mode_idx, i, train_logits.shape[0])
        valid_lambdas = load_arrays(
            dir, "valid", "lambdas", mode_idx, i, valid_logits.shape[0])
        test_lambdas = load_arrays(
            dir, "test", "lambdas", mode_idx, i, test_logits.shape[0])
        train_lambdas_list.append(train_lambdas)
        valid_lambdas_list.append(valid_lambdas)
        test_lambdas_list.append(test_lambdas)
        # lambdas (sync between features and contexts)
        train_lambdas = load_arrays(
            f_dir, "train", "lambdas", mode_idx, i, train_logits.shape[0])
        train_lambdas_checker2.append(train_lambdas)
        # features (for context)
        train_features = load_arrays(
            f_dir, "train", "features", mode_idx, i, train_logits.shape[0])
        valid_features = load_arrays(
            f_dir, "valid", "features", mode_idx, i, valid_logits.shape[0])
        test_features = load_arrays(
            f_dir, "test", "features", mode_idx, i, test_logits.shape[0])
        train_Bfeatures_list.append(train_features)
        valid_Bfeatures_list.append(valid_features)
        test_Bfeatures_list.append(test_features)
        # features for adversarial images
        train_supples = load_arrays(
            dir, "train", "advfeatures", mode_idx, i)
        valid_supples = load_arrays(
            dir, "valid", "advfeatures", mode_idx, i)
        test_supples = load_arrays(
            dir, "test", "advfeatures", mode_idx, i)
        train_Bsupples_list.append(train_supples)
        valid_Bsupples_list.append(valid_supples)
        test_Bsupples_list.append(test_supples)

    def putinA(mode_idx, i):
        # logits
        train_logits = load_arrays(
            dir, "train", "features", mode_idx, i)
        valid_logits = load_arrays(
            dir, "valid", "features", mode_idx, i)
        test_logits = load_arrays(
            dir, "test", "features", mode_idx, i)
        train_Alogits_list.append(train_logits)
        valid_Alogits_list.append(valid_logits)
        test_Alogits_list.append(test_logits)
        # lambdas (for sync)
        train_lambdas = load_arrays(
            dir, "train", "lambdas", mode_idx, i, train_logits.shape[0])
        train_lambdas_checker.append(train_lambdas)
        # features (for context)
        train_features = load_arrays(
            f_dir, "train", "features", mode_idx, i, train_logits.shape[0])
        valid_features = load_arrays(
            f_dir, "valid", "features", mode_idx, i, valid_logits.shape[0])
        test_features = load_arrays(
            f_dir, "test", "features", mode_idx, i, test_logits.shape[0])
        train_Afeatures_list.append(train_features)
        valid_Afeatures_list.append(valid_features)
        test_Afeatures_list.append(test_features)
        # features for adversarial images
        train_supples = load_arrays(
            dir, "train", "advfeatures", mode_idx, i)
        valid_supples = load_arrays(
            dir, "valid", "advfeatures", mode_idx, i)
        test_supples = load_arrays(
            dir, "test", "advfeatures", mode_idx, i)
        train_Asupples_list.append(train_supples)
        valid_Asupples_list.append(valid_supples)
        test_Asupples_list.append(test_supples)

    if config.pm060704 > 0:
        mode_idx = 0
        putinB(mode_idx, 0)
        putinA(mode_idx, config.pm060704)
    else:
        for mode_idx in range(1+n_Amodes):  # total 1+n_Amodes
            if mode_idx == 0:  # p_B
                for i in tqdm(range(n_samples_each_Bmode)):
                    putinB(mode_idx, i)
            else:  # p_A (mixture of modes)
                for i in tqdm(range(n_samples_each_Amode)):
                    putinA(mode_idx, i)

    train_logitsA = np.concatenate(train_Alogits_list, axis=0)
    train_logitsA = jnp.array(train_logitsA)
    del train_Alogits_list

    train_logitsB = np.concatenate(train_Blogits_list, axis=0)
    train_logitsB = jnp.array(train_logitsB)
    del train_Blogits_list

    train_featuresA = np.concatenate(train_Afeatures_list, axis=0)
    train_featuresA = jnp.array(train_featuresA)
    del train_Afeatures_list

    train_featuresB = np.concatenate(train_Bfeatures_list, axis=0)
    train_featuresB = jnp.array(train_featuresB)
    del train_Bfeatures_list

    train_supplesA = np.concatenate(train_Asupples_list, axis=0)
    train_supplesA = jnp.array(train_supplesA)
    del train_Asupples_list

    train_supplesB = np.concatenate(train_Bsupples_list, axis=0)
    train_supplesB = jnp.array(train_supplesB)
    del train_Bsupples_list

    valid_logitsA = np.concatenate(valid_Alogits_list, axis=0)
    valid_logitsA = jnp.array(valid_logitsA)
    del valid_Alogits_list

    valid_logitsB = np.concatenate(valid_Blogits_list, axis=0)
    valid_logitsB = jnp.array(valid_logitsB)
    del valid_Blogits_list

    valid_featuresA = np.concatenate(valid_Afeatures_list, axis=0)
    valid_featuresA = jnp.array(valid_featuresA)
    del valid_Afeatures_list

    valid_featuresB = np.concatenate(valid_Bfeatures_list, axis=0)
    valid_featuresB = jnp.array(valid_featuresB)
    del valid_Bfeatures_list

    valid_supplesA = np.concatenate(valid_Asupples_list, axis=0)
    valid_supplesA = jnp.array(valid_supplesA)
    del valid_Asupples_list

    valid_supplesB = np.concatenate(valid_Bsupples_list, axis=0)
    valid_supplesB = jnp.array(valid_supplesB)
    del valid_Bsupples_list

    test_logitsA = np.concatenate(test_Alogits_list, axis=0)
    test_logitsA = jnp.array(test_logitsA)
    del test_Alogits_list

    test_logitsB = np.concatenate(test_Blogits_list, axis=0)
    test_logitsB = jnp.array(test_logitsB)
    del test_Blogits_list

    test_featuresA = np.concatenate(test_Afeatures_list, axis=0)
    test_featuresA = jnp.array(test_featuresA)
    del test_Afeatures_list

    test_featuresB = np.concatenate(test_Bfeatures_list, axis=0)
    test_featuresB = jnp.array(test_featuresB)
    del test_Bfeatures_list

    test_supplesA = np.concatenate(test_Asupples_list, axis=0)
    test_supplesA = jnp.array(test_supplesA)
    del test_Asupples_list

    test_supplesB = np.concatenate(test_Bsupples_list, axis=0)
    test_supplesB = jnp.array(test_supplesB)
    del test_Bsupples_list

    train_lambdas = np.concatenate(train_lambdas_list, axis=0)
    train_lambdas = jnp.array(train_lambdas)
    del train_lambdas_list

    train_lambdas_checker = np.concatenate(train_lambdas_checker, axis=0)
    train_lambdas_checker = jnp.array(train_lambdas_checker)
    assert jnp.all(train_lambdas_checker == train_lambdas)
    del train_lambdas_checker

    train_lambdas_checker2 = np.concatenate(train_lambdas_checker2, axis=0)
    train_lambdas_checker2 = jnp.array(train_lambdas_checker2)
    assert jnp.all(train_lambdas_checker2 == train_lambdas)
    del train_lambdas_checker2

    valid_lambdas = np.concatenate(valid_lambdas_list, axis=0)
    valid_lambdas = jnp.array(valid_lambdas)
    del valid_lambdas_list

    test_lambdas = np.concatenate(test_lambdas_list, axis=0)
    test_lambdas = jnp.array(test_lambdas)
    del test_lambdas_list

    # classifying labels
    trn_labels = np.load(os.path.join(
        config.data_root, f'{config.data_name}/train_labels.npy'))
    tst_labels = np.load(os.path.join(
        config.data_root, f'{config.data_name}/test_labels.npy'))
    if config.data_name == "CIFAR10_x32":
        trn_labels, val_labels = trn_labels[:40960], trn_labels[40960:]
        num_classes = 10
    elif config.data_name == "CIFAR100_x32":
        trn_labels, val_labels = trn_labels[:40960], trn_labels[40960:]
        num_classes = 100

    _, repeats = get_info_in_dir(dir)
    trn_labels = jnp.tile(trn_labels, [n_samples_each_Bmode*repeats])
    val_labels = jnp.tile(val_labels, [n_samples_each_Bmode])
    tst_labels = jnp.tile(tst_labels, [n_samples_each_Bmode])

    if config.am051101:
        train_logitsA = get_avg_logits([train_logitsB, train_logitsA])
        valid_logitsA = get_avg_logits([valid_logitsB, valid_logitsA])
        test_logitsA = get_avg_logits([test_logitsB, test_logitsA])
        train_supplesA = get_avg_logits([train_supplesB, train_supplesA])
        valid_supplesA = get_avg_logits([valid_supplesB, valid_supplesA])
        test_supplesA = get_avg_logits([test_supplesB, test_supplesA])
    elif config.pm051010:
        train_logitsA = get_ens_logits([train_logitsB, train_logitsA])
        valid_logitsA = get_ens_logits([valid_logitsB, valid_logitsA])
        test_logitsA = get_ens_logits([test_logitsB, test_logitsA])
        train_supplesA = get_ens_logits([train_supplesB, train_supplesA])
        valid_supplesA = get_ens_logits([valid_supplesB, valid_supplesA])
        test_supplesA = get_ens_logits([test_supplesB, test_supplesA])

    def mix_dataset(rng, train_logits, valid_logits):
        assert len(train_logits) == len(valid_logits)
        train_size = train_logits[0].shape[0]
        valid_size = valid_logits[0].shape[0]
        perm_order = jax.random.permutation(
            rng, jnp.arange(train_size+valid_size))
        new_train_logits = []
        new_valid_logits = []
        for train, valid in zip(train_logits, valid_logits):
            assert train.shape[0] == train_size
            assert valid.shape[0] == valid_size
            total = jnp.concatenate([train, valid], axis=0)
            mixed_total = total[perm_order]
            new_train_logits.append(mixed_total[:train_size])
            new_valid_logits.append(mixed_total[train_size:])

        return new_train_logits, new_valid_logits

    if config.take_valid:
        (
            (
                train_logitsB,
                train_logitsA,
                train_featuresB,
                train_featuresA,
                trn_labels,
                train_lambdas,
                train_supplesB,
                train_supplesA
            ),
            (
                valid_logitsB,
                valid_logitsA,
                valid_featuresB,
                valid_featuresA,
                val_labels,
                valid_lambdas,
                valid_supplesB,
                valid_supplesA
            )
        ) = mix_dataset(
            rng,
            [
                train_logitsB,
                train_logitsA,
                train_featuresB,
                train_featuresA,
                trn_labels,
                train_lambdas,
                train_supplesB,
                train_supplesA
            ],
            [
                valid_logitsB,
                valid_logitsA,
                valid_featuresB,
                valid_featuresA,
                val_labels,
                valid_lambdas,
                valid_supplesB,
                valid_supplesA
            ]
        )

    def breakdown(rng, *args):
        output = []
        length = len(args[0])
        boundary = length//8
        indices = jax.random.permutation(rng, jnp.arange(0, length))
        train_indices, valid_indices = jnp.split(indices, [boundary])
        for a in args:
            train_valid = a[train_indices], a[valid_indices]
            output.append(train_valid)

        return tuple(output)

    if config.trainset_only:
        (
            (train_logitsB, valid_logitsB),
            (train_logitsA, valid_logitsA),
            (train_featuresB, valid_featuresB),
            (train_featuresA, valid_featuresA),
            (trn_labels, val_labels),
            (train_lambdas, valid_lambdas),
            (train_supplesB, valid_supplesB),
            (train_supplesA, valid_supplesA)
        ) = breakdown(rng,
                      train_logitsB,
                      train_logitsA,
                      train_featuresB,
                      train_featuresA,
                      trn_labels,
                      train_lambdas,
                      train_supplesB,
                      train_supplesA)

    if config.validset_only:
        (
            (train_logitsB, valid_logitsB),
            (train_logitsA, valid_logitsA),
            (train_featuresB, valid_featuresB),
            (train_featuresA, valid_featuresA),
            (trn_labels, val_labels),
            (train_lambdas, valid_lambdas),
            (train_supplesB, valid_supplesB),
            (train_supplesA, valid_supplesA)
        ) = breakdown(rng,
                      valid_logitsB,
                      valid_logitsA,
                      valid_featuresB,
                      valid_featuresA,
                      val_labels,
                      valid_lambdas,
                      valid_supplesB,
                      valid_supplesA)

    def get_conflict(logitsB, logitsA):
        probB = jax.nn.softmax(logitsB, axis=-1)
        probA = jax.nn.softmax(logitsA, axis=-1)
        predB = jnp.argmax(probB, axis=-1)
        predA = jnp.argmax(probA, axis=-1)
        conflicts = jnp.not_equal(predB, predA)
        return conflicts

    if config.conflict:
        train_conflicts = get_conflict(train_logitsB, train_logitsA)
        valid_conflicts = get_conflict(valid_logitsB, valid_logitsA)
        test_conflicts = get_conflict(test_logitsB, test_logitsA)

    def get_tfpn(logitsB, logitsA, labels):
        logpB = jax.nn.log_softmax(logitsB, axis=-1)
        logpA = jax.nn.log_softmax(logitsA, axis=-1)
        corrB = evaluate_acc(logpB, labels, log_input=True, reduction="none")
        corrA = evaluate_acc(logpA, labels, log_input=True, reduction="none")
        corrB = jnp.asarray(corrB, dtype=bool)
        corrA = jnp.asarray(corrA, dtype=bool)
        tfpn = dict(
            tp=jnp.logical_and(corrB, corrA),
            fp=jnp.logical_and(~corrB, corrA),
            fn=jnp.logical_and(corrB, ~corrA),
            tn=jnp.logical_and(~corrB, ~corrA)
        )
        tfpn_count = dict(
            tp=tfpn["tp"].sum(),
            fp=tfpn["fp"].sum(),
            fn=tfpn["fn"].sum(),
            tn=tfpn["tn"].sum(),
        )
        return tfpn, tfpn_count

    # TP/FP/FN/TN
    train_tfpn, train_tfpn_count = get_tfpn(
        train_logitsB, train_logitsA, trn_labels)
    valid_tfpn, valid_tfpn_count = get_tfpn(
        valid_logitsB, valid_logitsA, val_labels)
    test_tfpn, test_tfpn_count = get_tfpn(
        test_logitsB, test_logitsA, tst_labels)

    dataloaders = dict(
        train_length=len(train_logitsA),
        valid_length=len(valid_logitsA),
        test_length=len(test_logitsA),
        train_tfpn=train_tfpn_count,
        valid_tfpn=valid_tfpn_count,
        test_tfpn=test_tfpn_count,
        num_classes=num_classes,
        image_shape=(1, 32, 32, 3),
        trn_steps_per_epoch=math.ceil(len(train_logitsA)/config.optim_bs),
        val_steps_per_epoch=math.ceil(len(valid_logitsA)/config.optim_bs),
        tst_steps_per_epoch=math.ceil(len(test_logitsA)/config.optim_bs)
    )
    dataloaders["featureloader"] = partial(
        # _build_dataloader,
        _build_featureloader,
        images=train_logitsB,
        labels=train_logitsA,
        cls_labels=trn_labels,
        lambdas=train_lambdas,
        contexts=train_featuresB if not config.pm050905 else train_featuresA,
        conflicts=train_conflicts if config.conflict else None,
        tp=train_tfpn["tp"],
        fp=train_tfpn["fp"],
        fn=train_tfpn["fn"],
        tn=train_tfpn["tn"],
        supplesB=train_supplesB,
        supplesA=train_supplesA,
        batch_size=config.optim_bs,
        shuffle=True,
        transform=None
    )

    dataloaders["trn_featureloader"] = partial(
        # _build_dataloader,
        _build_featureloader,
        images=train_logitsB,
        labels=train_logitsA,
        cls_labels=trn_labels,
        lambdas=train_lambdas,
        contexts=train_featuresB if not config.pm050905 else train_featuresA,
        conflicts=train_conflicts if config.conflict else None,
        tp=train_tfpn["tp"],
        fp=train_tfpn["fp"],
        fn=train_tfpn["fn"],
        tn=train_tfpn["tn"],
        supplesB=train_supplesB,
        supplesA=train_supplesA,
        batch_size=config.optim_bs,
        shuffle=False,
        transform=None
    )
    dataloaders["val_featureloader"] = partial(
        # _build_dataloader,
        _build_featureloader,
        images=valid_logitsB,
        labels=valid_logitsA,
        cls_labels=val_labels,
        lambdas=valid_lambdas,
        contexts=valid_featuresB if not config.pm050905 else valid_featuresA,
        conflicts=valid_conflicts if config.conflict else None,
        tp=valid_tfpn["tp"],
        fp=valid_tfpn["fp"],
        fn=valid_tfpn["fn"],
        tn=valid_tfpn["tn"],
        supplesB=valid_supplesB,
        supplesA=valid_supplesA,
        batch_size=config.optim_bs,
        shuffle=False,
        transform=None
    )
    dataloaders["tst_featureloader"] = partial(
        # _build_dataloader,
        _build_featureloader,
        images=test_logitsB,
        labels=test_logitsA,
        cls_labels=tst_labels,
        lambdas=test_lambdas,
        contexts=test_featuresB if not config.pm050905 else test_featuresA,
        conflicts=test_conflicts if config.conflict else None,
        tp=test_tfpn["tp"],
        fp=test_tfpn["fp"],
        fn=test_tfpn["fn"],
        tn=test_tfpn["tn"],
        supplesB=test_supplesB,
        supplesA=test_supplesA,
        batch_size=config.optim_bs,
        shuffle=False,
        transform=None
    )
    normalize = partial(normalize_logits,  features_dir=dir)
    unnormalize = partial(unnormalize_logits,  features_dir=dir)
    f_normalize = partial(
        normalize_logits,  features_dir=f_dir) if config.context else None
    f_unnormalize = partial(
        unnormalize_logits,  features_dir=f_dir) if config.context else None

    if config.get_stats:
        # ------------------------------------------------------------------
        #  Compute statistics (mean, std)
        # ------------------------------------------------------------------
        count = 0
        _sum = 0
        for batch in dataloaders["trn_featureloader"](rng=None):
            logitsB = batch["images"]
            logitsA = batch["labels"]
            # 0: pmap dimension, 1: batch dimension
            _sum += jnp.sum(logitsB, [0, 1])+jnp.sum(logitsA, [0, 1])
            count += 2*batch["marker"].sum()
        mean = _sum/count
        _sum = 0
        for batch in dataloaders["trn_featureloader"](rng=None):
            logitsB = batch["images"]
            logitsA = batch["labels"]
            marker = expand_to_broadcast(batch["marker"], logitsB, axis=2)
            mseB = jnp.where(marker, (logitsB-mean)**2,
                             jnp.zeros_like(logitsB))
            mseA = jnp.where(marker, (logitsA-mean)**2,
                             jnp.zeros_like(logitsA))
            _sum += jnp.sum(mseB, axis=[0, 1])
            _sum += jnp.sum(mseA, axis=[0, 1])
        var = _sum/count
        std = jnp.sqrt(var)
        print("dims", mean.shape)
        print("mean", mean)
        print("std", std)
        assert False, "Statistics are Calculated"

    def get_rate(div):
        if div == "train":
            loader = dataloaders["trn_featureloader"](rng=None)
            length = dataloaders["train_length"]
        elif div == "valid":
            loader = dataloaders["val_featureloader"](rng=None)
            length = dataloaders["valid_length"]
        elif div == "test":
            loader = dataloaders["tst_featureloader"](rng=None)
            length = dataloaders["test_length"]
        count = 0
        for batch in loader:
            logitsB = batch["images"]
            logitsA = batch["labels"]
            predB = jax.nn.log_softmax(logitsB, axis=-1)
            predA = jax.nn.log_softmax(logitsA, axis=-1)
            predB = jnp.argmax(predB, axis=-1)
            predA = jnp.argmax(predA, axis=-1)
            conflicts = jnp.where(
                batch["marker"],
                jnp.not_equal(predB, predA),
                jnp.zeros_like(count))
            count += jnp.sum(conflicts)
        count /= length
        return count

    if config.get_conflict_rate:
        print("train", get_rate("train"))
        print("valid", get_rate("valid"))
        print("test", get_rate("test"))
        assert False, "Conflict Rate is Calculated"

    return dataloaders, normalize, unnormalize, f_normalize, f_unnormalize


class TrainState(train_state.TrainState):
    batch_stats: Any
    rng: Any
    dynamic_scale: dynamic_scale_lib.DynamicScale
    ema_params: Any
    eps: jnp.float32

    n_T: jnp.int32

    def forward(self, x0, x1, training=True, **kwargs):
        # rng
        assert 'rng' in kwargs.keys()
        rng = kwargs["rng"]
        _ts = jax.random.uniform(
            rng, (x0.shape[0],), minval=self.eps, maxval=1.)  # (B,)
        # time 0 --> x0, time 1 --> x1
        x_t = batch_mul((1 - _ts), x0) + batch_mul(_ts, x1)

        kwargs["t"] = _ts
        kwargs["training"] = training
        return x_t, kwargs

    def sample(self, apply, x0, ctx, cfl, eps, n_T):
        shape = x0.shape
        batch_size = shape[0]
        x_n = x0  # (B, d)

        timesteps = jnp.linspace(eps, 1., n_T)
        timesteps = jnp.concatenate([jnp.array([0]), timesteps], axis=0)

        def body_fn(n, val):
            """
              n in [0, self.n_T - 1]
            """
            x_n = val
            current_t = jnp.array([timesteps[n_T - n]])
            next_t = jnp.array([timesteps[n_T - n - 1]])
            current_t = jnp.tile(current_t, [batch_size])
            next_t = jnp.tile(next_t, [batch_size])

            eps = apply(x_n, current_t, ctx, cfl)

            x_n = x_n + batch_mul(next_t - current_t, eps)

            return x_n

        x_n = jax.lax.fori_loop(0, n_T, body_fn, x_n)

        return x_n


def launch(config, print_fn):
    rng = jax.random.PRNGKey(config.seed)
    init_rng, data_rng, rng = jax.random.split(rng, 3)

    if config.save_cls2:
        ckpt = checkpoints.restore_checkpoint(
            ckpt_dir=config.save_cls2,
            target=None
        )
        for k, v in ckpt["config"].items():
            setattr(config, k, v)

    if config.am050911:
        config.gamma = 1
        config.start_rglr = 10
        config.stop_grad = True
        config.corrector = 1
        config.rglr_list = "kld"
        config.context = True
        config.sample_ensemble = 20
        config.p2_weight = True
        config.determ_eps = True
    if config.pm050902:
        config.gamma = 1
        config.start_rglr = 10
        config.stop_grad = True
        config.corrector = 1
        config.rglr_list = "kld"
        config.context = True
        config.sample_ensemble = 20
        config.p2_weight = True
        config.determ_eps = True
        # config.p2_gamma = True
        # config.p2_gammainv = True

    if config.features_dir in logit_dir_list:
        if config.data_name == "CIFAR100_x32":
            x_dim = 100
        else:
            x_dim = 10
    elif config.features_dir in feature_dir_list:
        if config.data_name == "CIFAR100_x32":
            x_dim = 128
        else:
            x_dim = 64
    elif config.features_dir in feature2_dir_list:
        if config.data_name == "CIFAR100_x32":
            x_dim = (8, 8, 128)
        else:
            x_dim = (8, 8, 64)
    if config.contexts_dir in feature_dir_list:
        if config.data_name == "CIFAR100_x32":
            ctx_dim = 128
        else:
            ctx_dim = 64
    elif config.contexts_dir in feature2_dir_list:
        if config.data_name == "CIFAR100_x32":
            ctx_dim = (8, 8, 128)
        else:
            ctx_dim = (8, 8, 64)
    logit_dim = None
    if config.supple:
        logit_dim = x_dim
        x_dim = x_dim*2
    # specify precision
    model_dtype = jnp.float32
    if config.precision == 'fp16':
        model_dtype = jnp.bfloat16 if jax.local_devices(
        )[0].platform == 'tpu' else jnp.float16

    # build dataloaders
    (
        dataloaders, normalize, unnormalize, f_normalize, f_unnormalize
    ) = build_featureloaders(
        config, data_rng)
    config.n_classes = dataloaders["num_classes"]

    if isinstance(x_dim, int):
        score_func = FeatureUnet(
            in_channels=x_dim,
            ver=config.version,
            n_feat=config.n_feat,
            droprate=config.droprate,
            context=ctx_dim if config.context else None,
            conflict=config.conflict
        )
        classifier = FlaxResNetClassifier(
            num_classes=dataloaders["num_classes"]
        )
    elif isinstance(x_dim, tuple) and len(x_dim) > 1:
        image_size = x_dim[0]
        num_channels = config.n_feat
        channel_mult = ""
        learn_sigma = config.learn_sigma
        num_res_blocks = config.num_res_blocks
        class_cond = False
        attention_resolutions = "16"
        num_heads = config.num_heads
        num_head_channels = -1
        num_heads_upsample = -1
        use_scale_shift_norm = config.use_scale_shift_norm
        dropout = config.droprate
        resblock_updown = config.resblock_updown
        use_new_attention_order = config.use_new_attention_order
        in_channels = x_dim[-1]
        if channel_mult == "":
            if image_size == 512:
                channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
            elif image_size == 256:
                channel_mult = (1, 1, 2, 2, 4, 4)
            elif image_size == 128:
                channel_mult = (1, 1, 2, 3, 4)
            elif image_size == 64:
                channel_mult = (1, 2, 3, 4)
            elif image_size == 32:
                channel_mult = (1, 2, 4)
            elif image_size == 16:
                channel_mult = (1, 2, 4)
            elif image_size == 8:
                channel_mult = (1, 2)
            else:
                raise ValueError(f"unsupported image size: {image_size}")
        else:
            channel_mult = tuple(int(ch_mult)
                                 for ch_mult in channel_mult.split(","))

        attention_ds = []
        for res in attention_resolutions.split(","):
            attention_ds.append(image_size // int(res))

        score_func = UNetModel(
            image_size=image_size,
            in_channels=in_channels,
            model_channels=num_channels,
            out_channels=(in_channels if not learn_sigma else 2*in_channels),
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=dropout,
            channel_mult=channel_mult,
            num_classes=(dataloaders["num_classes"] if class_cond else None),
            dtype=model_dtype,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            context=ctx_dim if config.context else None
        )
        classifier = FlaxResNetClassifier2(
            num_classes=dataloaders["num_classes"]
        )

    corrector = CorrectionModel(layers=config.corrector)

    # initialize model

    def initialize_model(key, model, input_dim=None):
        input_dim = input_dim or x_dim
        init_t = jnp.empty((1,))
        if isinstance(x_dim, int):
            init_x = jnp.empty((1, input_dim), model_dtype)
        else:
            init_x = jnp.empty((1, *input_dim), model_dtype)
        if isinstance(ctx_dim, int):
            init_ctx = jnp.empty((1, ctx_dim))
        else:
            init_ctx = jnp.empty((1, *ctx_dim))
        init_cfl = jnp.empty((1,))
        return model.init(
            {'params': key},
            x=init_x,
            t=init_t,
            ctx=init_ctx,
            cfl=init_cfl,
            training=False
        )
    variables = initialize_model(init_rng, score_func)
    variables_clsB = initialize_model(init_rng, classifier)
    variables_clsA = initialize_model(init_rng, classifier)
    variables_cor = initialize_model(init_rng, corrector, input_dim=logit_dim)

    def load_classifier(variables_cls, ckpt_dir):
        variables_cls = variables_cls.unfreeze()
        ckpt = checkpoints.restore_checkpoint(
            ckpt_dir=os.path.join(ckpt_dir, "sghmc"),
            target=None
        )
        params = ckpt["model"]["params"].get("Dense_0")
        if params is None:
            params = ckpt["model"]["params"]["head"]
        variables_cls["params"]["Dense_0"] = params
        variables_cls = freeze(variables_cls)
        del ckpt

        return variables_cls

    def correct_feature(cparams, feature):
        return corrector.apply({"params": cparams}, feature)

    name = config.data_name
    style = config.model_style
    shared = config.shared_head
    variables_clsB = load_classifier(
        variables_clsB, model_list(name, style, shared)[0])  # current mode
    variables_clsA = load_classifier(
        variables_clsA, model_list(name, style, shared)[1])  # target mode

    dynamic_scale = None
    if config.precision == 'fp16' and jax.local_devices()[0].platform == 'gpu':
        dynamic_scale = dynamic_scale_lib.DynamicScale()

    total_epochs = config.optim_ne + config.finetune_ne

    # define optimizer with scheduler
    scheduler = optax.cosine_decay_schedule(
        init_value=config.optim_lr,
        decay_steps=total_epochs * dataloaders["trn_steps_per_epoch"])
    optimizer = optax.adamw(
        learning_rate=scheduler, weight_decay=config.optim_weight_decay)
    if isinstance(score_func, UNetModel):
        partition_optimizer = {"trainable": optimizer,
                               "frozen": optax.set_to_zero()}
        param_partitions = freeze(flax.traverse_util.path_aware_map(
            lambda path, _: "frozen" if "frozen" in path else "trainable", variables["params"]))
        optimizer = optax.multi_transform(
            partition_optimizer, param_partitions)

    # TODO 2: change TrainState to rectified flow TrainState.
    # Train state of diffusion bridge
    state = TrainState.create(
        apply_fn=score_func.apply,
        params=(variables["params"], variables_cor["params"]),
        ema_params=(variables["params"], variables_cor["params"]),
        tx=optimizer,
        batch_stats=variables.get("batch_stats"),
        rng=init_rng,
        dynamic_scale=dynamic_scale,
        n_T=config.T,
        eps=config.eps)

    if config.save_cls2:
        ckpt = dict(model=state, config=dict(),
                    best_acc=ckpt["best_acc"], examples=ckpt["examples"])
        checkpoints.restore_checkpoint(
            ckpt_dir=config.save_cls2, target=ckpt)
        state = ckpt["model"]

    # gamma schedule
    if config.gamma > 0 and config.start_rglr > 0:
        gamma_schedules = [optax.constant_schedule(
            0.), optax.constant_schedule(config.gamma)]
        gamma_boundaries = [config.start_rglr *
                            dataloaders["trn_steps_per_epoch"]]
        gamma_fn = optax.join_schedules(
            schedules=gamma_schedules,
            boundaries=gamma_boundaries
        )
    else:
        def gamma_fn(step): return config.gamma

    # objective
    def mse_loss(noise, output, lamb=None, mask=None):
        # assert len(output.shape) == 2
        sum_axis = list(range(1, len(output.shape[1:])+1))
        if lamb is None:
            lamb = 1
        else:
            lamb = expand_to_broadcast(lamb, output, axis=1)
        if mask is not None:
            mask = 5*mask + (1-mask)
            noise = mask*noise
            output = mask*output
        loss = jnp.sum(lamb * jnp.abs(noise-output) **
                       config.mse_power, axis=sum_axis)
        return loss

    def get_logits(feat, mode, cparams=None):
        if mode.lower() == "c":
            if config.stop_grad:
                feat = jax.lax.stop_gradient(feat)
            if config.corrector > 0:
                feat = correct_feature(cparams, feat)
        if "last" in config.features_dir:
            if mode.lower() == "a" or mode.lower() == "c":
                variables_cls_params = variables_clsA["params"]
            elif mode.lower() == "b":
                variables_cls_params = variables_clsB["params"]
            logits = classifier.apply(
                {"params": variables_cls_params}, feat
            )
        else:
            logits = feat
        return logits

    def ensemble_accuracy(f_list, labels, marker, mode, cparams=None):
        avg_preds = 0
        each_acc = []
        for m, f_gen in zip(mode, f_list):
            logits = get_logits(f_gen, m, cparams)

            predictions = jax.nn.log_softmax(logits, axis=-1)
            avg_preds += jnp.exp(predictions)
            acc = evaluate_acc(
                predictions, labels, log_input=True, reduction="none")
            nll = evaluate_nll(
                predictions, labels, log_input=True, reduction='none')
            acc = jnp.sum(jnp.where(marker, acc, jnp.zeros_like(acc)))
            nll = jnp.sum(jnp.where(marker, nll, jnp.zeros_like(nll)))
            each_acc.append((acc, nll))
        avg_preds /= len(f_list)

        acc = evaluate_acc(
            avg_preds, labels, log_input=False, reduction="none")
        nll = evaluate_nll(
            avg_preds, labels, log_input=False, reduction='none')
        acc = jnp.sum(jnp.where(marker, acc, jnp.zeros_like(acc)))
        nll = jnp.sum(jnp.where(marker, nll, jnp.zeros_like(nll)))
        return (acc, nll), each_acc

    def ensemble_top2accuracy(f_list, labels, marker, mode, cparams=None):
        avg_preds = 0
        each_acc = []
        for m, f_gen in zip(mode, f_list):
            logits = get_logits(f_gen, m, cparams)
            predictions = jax.nn.log_softmax(logits, axis=-1)
            avg_preds += jnp.exp(predictions)
            acc = evaluate_top2acc(
                predictions, labels, log_input=True, reduction="none")
            acc = jnp.sum(jnp.where(marker, acc, jnp.zeros_like(acc)))
            each_acc.append(acc)
        avg_preds /= len(f_list)

        acc = evaluate_top2acc(
            avg_preds, labels, log_input=False, reduction="none")
        acc = jnp.sum(jnp.where(marker, acc, jnp.zeros_like(acc)))
        return acc, each_acc

    def ensemble_topNaccuracy(top, f_list, labels, marker, mode, cparams=None):
        avg_preds = 0
        each_acc = []
        for m, f_gen in zip(mode, f_list):
            logits = get_logits(f_gen, m, cparams)
            predictions = jax.nn.log_softmax(logits, axis=-1)
            avg_preds += jnp.exp(predictions)
            acc = evaluate_topNacc(
                predictions, labels, top, log_input=True, reduction="none")
            acc = jnp.sum(jnp.where(marker, acc, jnp.zeros_like(acc)))
            each_acc.append(acc)
        avg_preds /= len(f_list)

        acc = evaluate_topNacc(
            avg_preds, labels, top, log_input=False, reduction="none")
        acc = jnp.sum(jnp.where(marker, acc, jnp.zeros_like(acc)))
        return acc, each_acc

    def ensemble_tfpn_accuracy(f_list, labels, marker, tfpn, mode, cparams=None):
        avg_preds = 0
        each_acc = []
        for m, f_gen in zip(mode, f_list):
            logits = get_logits(f_gen, m, cparams)

            predictions = jax.nn.log_softmax(logits, axis=-1)
            avg_preds += jnp.exp(predictions)
            acc = evaluate_acc(
                predictions, labels, log_input=True, reduction="none")
            tfpn_acc = dict()
            for k, v in tfpn.items():
                v = jnp.logical_and(v, marker)
                tfpn_acc[k] = jnp.sum(jnp.where(v, acc, jnp.zeros_like(acc)))
            each_acc.append(tfpn_acc)
        avg_preds /= len(f_list)

        acc = evaluate_acc(
            avg_preds, labels, log_input=False, reduction="none")
        tfpn_acc = dict()
        for k, v in tfpn.items():
            v = jnp.logical_and(v, marker)
            tfpn_acc[k] = jnp.sum(jnp.where(v, acc, jnp.zeros_like(acc)))
        return tfpn_acc, each_acc

    def cross_entropy(f_gen, labels, marker, mode="C", cparams=None):
        logits = get_logits(f_gen, mode, cparams)
        target = common_utils.onehot(
            labels, num_classes=logits.shape[-1])  # [B, K,]
        loss = -jnp.sum(
            target * jax.nn.log_softmax(logits, axis=-1),
            axis=-1)      # [B,]
        loss = jnp.where(marker, loss, 0)
        return loss

    def distill_entropy(logitsA_eps, logitsA, marker, mode_eps="C", mode="A", cparams=None):
        logitsA_eps = get_logits(logitsA_eps, mode_eps, cparams)
        logitsA = get_logits(logitsA, mode, cparams)
        soft_labels = jax.nn.softmax(logitsA, axis=-1)
        soft_preds = jax.nn.log_softmax(logitsA_eps, axis=-1)
        loss = -jnp.sum(soft_labels*soft_preds, axis=-1)
        loss = jnp.where(marker, loss, 0)
        return loss

    def ensemble_entropy(logits, labels, marker, mode=["B", "C"], cparams=None):
        assert len(logits) == len(mode)
        ens_preds = 0
        for m, _logits in zip(mode, logits):
            logits = get_logits(_logits, m, cparams)
            ens_preds += jnp.exp(jax.nn.log_softmax(logits, axis=-1))
        ens_preds /= len(ens_preds)
        target = common_utils.onehot(
            labels, num_classes=logits.shape[-1])  # [B, K,]
        loss = -jnp.sum(
            target * jnp.log(ens_preds),
            axis=-1)
        loss = jnp.where(marker, loss, 0)
        return loss

    def kl_divergence(logit_tar, logit_ref, marker, mode_tar="C", mode_ref="A", cparams=None, mask=None):
        """
        logit_tar ~ q(x)
        logit_ref ~ p(x)

        return KL(q||p)
        """

        logit_tar = get_logits(logit_tar, mode_tar, cparams)
        logit_ref = get_logits(logit_ref, mode_ref, cparams)
        logq = jax.nn.log_softmax(logit_tar, axis=-1)
        logp = jax.nn.log_softmax(logit_ref, axis=-1)
        if mask is not None:
            logq = mask*logq
        q = jnp.exp(logq)
        integrand = q*(logq-logp)
        assert len(integrand.shape) == 2
        kld = jnp.sum(integrand, axis=-1)
        kld = jnp.where(marker, kld, 0)
        return jnp.sum(kld)

    def ensemble_kld(logits_tar, logits_ref, marker, mode_tar="C", mode_ref="A", cparams=None):
        logits_tar = [get_logits(tar, mode_tar, cparams) for tar in logits_tar]
        logits_ref = [get_logits(ref, mode_ref, cparams) for ref in logits_ref]
        q_list = [jax.nn.softmax(tar, axis=-1) for tar in logits_tar]
        p_list = [jax.nn.softmax(ref, axis=-1) for ref in logits_ref]
        q = sum(q_list)/len(q_list)
        p = sum(p_list)/len(p_list)
        logq = jnp.log(q)
        logp = jnp.log(p)
        integrand = q*(logq-logp)
        kld = jnp.sum(integrand, axis=-1)
        kld = jnp.where(marker, kld, 0)
        return jnp.sum(kld)

    rglr_list = config.rglr_list.split(",")
    rglr_coef = [float(a) for a in config.rglr_coef.split(",")]

    # TODO 3: loss functions
    def collect_rglr(unnorm_x0eps, unnorm_logitsA, unnorm_logitsB, labels, batch, cparams=None):
        total_rglr = 0
        stable = 1e4
        power = config.mse_power
        for coef, name in zip(rglr_coef, rglr_list):
            if name == "":
                rglr = 0
            elif name == "distill":
                # good for val/loss
                rglr = 300 * distill_entropy(
                    unnorm_x0eps, unnorm_logitsA, batch["marker"], "C", "A", cparams)
            elif name == "cel2":  # good for val/kld in takevalid
                rglr = 4000*mse_loss(
                    cross_entropy(unnorm_x0eps, labels,
                                  batch["marker"], "C", cparams),
                    cross_entropy(unnorm_logitsA, labels, batch["marker"], "C", cparams))
            elif name == "kld":
                mask = 1-common_utils.onehot(
                    labels, num_classes=unnorm_x0eps.shape[-1]) if config.maxmask else None
                rglr = kl_divergence(
                    unnorm_x0eps, unnorm_logitsA, batch["marker"], "C", "A", cparams, mask)
            elif name == "dkld":
                rglr = 0.5*(
                    kl_divergence(
                        unnorm_x0eps, unnorm_logitsA, batch["marker"], "C", "A", cparams)
                    + kl_divergence(
                        unnorm_logitsA, unnorm_x0eps, batch["marker"], "C", "A", cparams)
                )
            elif name == "rkld":
                rglr = kl_divergence(
                    unnorm_logitsA, unnorm_x0eps, batch["marker"], "A", "C", cparams)  # good, sucks for takevalid
            elif name == "ce":  # sucks for takvalid
                rglr = 100*cross_entropy(
                    unnorm_x0eps, labels, batch["marker"], "C", cparams)
            elif name == "rdistill":
                rglr = 400 * distill_entropy(
                    unnorm_logitsA, unnorm_x0eps, batch["marker"], "A", "C", cparams)
            elif name == "ensemble":  # sucks for takevalid
                rglr = 100*ensemble_entropy(
                    [unnorm_logitsB, unnorm_x0eps], labels, batch["marker"], ["B", "C"], cparams)  # good for val/loss
            elif name == "skld":  # sucks for takevalid
                rglr = - 1e-6 * kl_divergence(unnorm_x0eps, unnorm_logitsB,
                                              batch["marker"], "C", "B", cparams)
            elif name == "rskld":  # sucks for takevalid
                rglr = - 1e-6 * kl_divergence(unnorm_logitsB, unnorm_x0eps,
                                              batch["marker"], "B", "C", cparams)
            elif name == "l2mse":
                rglr = mse_loss(
                    mse_loss(unnorm_logitsB, unnorm_x0eps),
                    mse_loss(unnorm_logitsB, unnorm_logitsA)
                )
            elif name == "l2rkld":
                rglr = mse_loss(
                    kl_divergence(unnorm_logitsA, unnorm_x0eps,
                                  batch["marker"], "A", "C", cparams),
                    kl_divergence(unnorm_logitsA, unnorm_logitsB,
                                  batch["marker"], "A", "B", cparams)
                )
            elif name == "minmse":
                rglr = -jnp.minimum(
                    mse_loss(unnorm_logitsB, unnorm_x0eps),
                    mse_loss(unnorm_logitsB, unnorm_logitsA)
                )
            elif name == "maxmse":
                rglr = jnp.maximum(
                    mse_loss(unnorm_logitsB, unnorm_x0eps),
                    mse_loss(unnorm_logitsB, unnorm_logitsA)
                )
            elif name == "var":
                p = jax.nn.softmax(unnorm_x0eps, axis=-1)
                indices = 1+jnp.arange(0, p.shape[-1])[None, ...]
                ECx2J = jnp.sum(indices**2*p, axis=-1)
                ECxJ2 = jnp.sum(indices*p, axis=-1)**2
                rglr = 100*(ECx2J - ECxJ2)
            elif name == "prob":
                p_C = stable*jax.nn.softmax(unnorm_x0eps, axis=-1)
                p_A = stable*jax.nn.softmax(unnorm_logitsA, axis=-1)
                rglr = mse_loss(p_C, p_A) / stable**power
            elif name == "regress":
                p_C = stable*jax.nn.softmax(unnorm_x0eps, axis=-1)
                p_y = stable*common_utils.onehot(labels, p_C.shape[-1])
                rglr = mse_loss(p_C, p_y) / stable**power
            elif name == "ymatch":
                p_C = stable*jax.nn.softmax(unnorm_x0eps, axis=-1)
                p_A = stable*jax.nn.softmax(unnorm_logitsA, axis=-1)
                y = common_utils.onehot(labels, p_C.shape[-1])
                rglr = mse_loss(y*p_C, y*p_A) / stable**power
            elif name == "maxmatch":
                p_C = stable*jax.nn.softmax(unnorm_x0eps, axis=-1)
                p_A = stable*jax.nn.softmax(unnorm_logitsA, axis=-1)
                p_B = stable*jax.nn.softmax(unnorm_logitsA, axis=-1)
                labelsA = jnp.argmax(p_A, axis=-1)
                labelsB = jnp.argmax(p_B, axis=-1)
                yA = common_utils.onehot(labelsA, p_A.shape[-1])
                yB = common_utils.onehot(labelsB, p_B.shape[-1])
                mask = jnp.logical_or(yA, yB)
                rglr = mse_loss(mask*p_C, mask*p_A) / stable**power
            elif name == "meanmatch":
                p_C = stable*jax.nn.softmax(unnorm_x0eps, axis=-1)
                p_A = stable*jax.nn.softmax(unnorm_logitsA, axis=-1)
                p_B = stable*jax.nn.softmax(unnorm_logitsB, axis=-1)
                p_A_mean = jnp.mean(p_A, axis=-1)[..., None]
                p_B_mean = jnp.mean(p_B, axis=-1)[..., None]
                maskA = p_A > p_A_mean
                maskB = p_B > p_B_mean
                mask = jnp.logical_or(maskA, maskB)
                rglr = mse_loss(mask*p_C, mask*p_A) / stable**power
            elif name == "conflict":
                p_C = stable*jax.nn.softmax(unnorm_x0eps, axis=-1)
                p_A = stable*jax.nn.softmax(unnorm_logitsA, axis=-1)
                p_B = stable*jax.nn.softmax(unnorm_logitsB, axis=-1)
                mask = jnp.argmax(p_A, axis=-1) != jnp.argmax(p_B, axis=-1)
                mask = mask[:, None]
                rglr = mse_loss(mask*p_C, mask*p_A) / stable**power
            else:
                raise Exception("Invalid Regularization Name")
            total_rglr += coef*rglr
        return total_rglr

    # training

    def step_train(state, batch, config):
        _logitsB = batch["images"]  # the current mode
        _logitsA = batch["labels"]  # mixture of other modes
        labels = batch["cls_labels"]
        logitsB = normalize(_logitsB)
        logitsA = normalize(_logitsA)
        contexts = f_normalize(batch["contexts"]) if config.context else None
        conflicts = batch["conflicts"] if config.conflict else None
        supplesB = normalize(batch["supplesB"]) if config.supple else None
        supplesA = normalize(batch["supplesA"]) if config.supple else None
        if config.supple:
            logitsB = jnp.concatenate([logitsB, supplesB], axis=-1)
            logitsA = jnp.concatenate([logitsA, supplesA], axis=-1)

        def loss_fn(params):
            params, cparams = params
            logits_t, kwargs = state.forward(
                logitsA, logitsB, training=True, rng=state.rng)  # linear interpolation of logitsA --> logitsB
            params_dict = dict(params=params)
            rngs_dict = dict(dropout=state.rng)
            mutable = []
            if state.batch_stats is not None:
                params_dict["batch_stats"] = state.batch_stats
                mutable.append("batch_stats")
            epsilon, new_model_state = state.apply_fn(
                params_dict, logits_t, ctx=contexts, cfl=conflicts,
                mutable=mutable, rngs=rngs_dict, **kwargs)  # epsilon: expected logitsB - logitsA
            # logits
            diff = logitsB - logitsA
            unnorm_logitsA = _logitsA
            unnorm_logitsB = _logitsB
            p2 = (lambda t: 0.0001 + 0.5*(1-0.0001)*(1+jnp.cos(math.pi *
                  jnp.where(t > 0.5, 2*t-1, 1-2*t))))(kwargs["t"])
            lamb = p2 if config.p2_weight else jnp.ones_like(kwargs["t"])
            mask = None
            if config.pm051101:
                p_A = jax.nn.softmax(unnorm_logitsA, axis=-1)
                p_B = jax.nn.softmax(unnorm_logitsB, axis=-1)
                mask = jnp.argmax(p_A, axis=-1) != jnp.argmax(p_B, axis=-1)
                mask = expand_to_broadcast(mask, diff, axis=1)
            elif config.pm053105:
                p_A = jax.nn.softmax(unnorm_logitsA, axis=-1)
                p_B = jax.nn.softmax(unnorm_logitsB, axis=-1)
                mask = jnp.argmax(p_A, axis=-1) == jnp.argmax(p_B, axis=-1)
                mask = expand_to_broadcast(mask, diff, axis=1)
            loss = mse_loss(epsilon, diff, lamb=lamb, mask=mask)
            x0eps = logits_t  # - _sigma_t*epsilon
            if config.supple:
                unnorm_x0eps = unnormalize(x0eps[..., :x_dim//2])
            else:
                unnorm_x0eps = unnormalize(x0eps)
            celoss = collect_rglr(
                unnorm_x0eps, unnorm_logitsA, unnorm_logitsB, labels, batch, cparams)
            # celoss /= sigma_t**2
            count = jnp.sum(batch["marker"])
            loss = jnp.where(batch["marker"], loss, jnp.zeros_like(loss))
            celoss = jnp.where(batch["marker"], celoss, jnp.zeros_like(celoss))
            if config.p2_gamma or config.p2_gammainv:
                if config.p2_gamma:
                    gamma = kwargs["t"] * gamma_fn(state.step)
                else:
                    gamma = (1-kwargs["t"]) * gamma_fn(state.step)
                totalloss = (1-gamma)*loss + gamma*celoss
                totalloss = jnp.sum(totalloss) / count
                loss = jnp.sum(loss) / count
                celoss = jnp.sum(celoss) / count
            elif config.beta > 0:
                loss = jnp.sum(loss) / count
                celoss = jnp.sum(celoss) / count
                totalloss = loss + config.beta*celoss
            else:
                loss = jnp.sum(loss) / count
                celoss = jnp.sum(celoss) / count
                gamma = gamma_fn(state.step)
                totalloss = (1-gamma)*loss + gamma*celoss

            metrics = OrderedDict(
                {"loss": loss*count, "count": count, "celoss": celoss*count})

            return totalloss, (metrics, new_model_state)

        dynamic_scale = state.dynamic_scale
        if dynamic_scale:
            dynamic_scale, is_fin, (loss, aux), grads = dynamic_scale.value_and_grad(
                loss_fn, has_aux=True, axis_name='batch')(state.params)
        else:
            (loss, aux), grads = jax.value_and_grad(
                loss_fn, has_aux=True)(state.params)
            grads = jax.lax.pmean(grads, axis_name='batch')

        metrics, new_model_state = aux
        new_state = state.apply_gradients(
            grads=grads, batch_stats=new_model_state.get('batch_stats'))
        a = config.ema_decay
        new_state = new_state.replace(
            ema_params=jax.tree_util.tree_map(
                lambda wt, ema_tm1: jnp.where(a < 1, a*wt + (1-a)*ema_tm1, wt),
                new_state.params,
                new_state.ema_params))
        metrics = jax.lax.psum(metrics, axis_name="batch")

        if dynamic_scale:
            new_state = new_state.replace(
                opt_state=jax.tree_util.tree_map(
                    partial(jnp.where, is_fin), new_state.opt_state, state.opt_state),
                params=jax.tree_util.tree_map(
                    partial(jnp.where, is_fin), new_state.params, state.params),
                dynamic_scale=dynamic_scale)

        return new_state, metrics

    def step_valid(state, batch):
        ema_params, _ = state.ema_params
        # normalize
        logitsB = batch["images"]  # the current mode
        logitsA = batch["labels"]  # mixture of other modes
        contexts = batch["contexts"]
        logitsB = normalize(logitsB)
        logitsA = normalize(logitsA)
        contexts = f_normalize(contexts) if config.context else None
        conflicts = batch["conflicts"] if config.conflict else None
        supplesB = normalize(batch["supplesB"]) if config.supple else None
        supplesA = normalize(batch["supplesA"]) if config.supple else None
        if config.supple:
            logitsB = jnp.concatenate([logitsB, supplesB], axis=-1)
            logitsA = jnp.concatenate([logitsA, supplesA], axis=-1)

        # get interpolation
        logits_t, kwargs = state.forward(
            logitsA, logitsB, training=False, rng=state.rng)
        # get epsilon
        params_dict = dict(params=ema_params)
        rngs_dict = dict(dropout=state.rng)
        if state.batch_stats is not None:
            params_dict["batch_stats"] = state.batch_stats
        output = state.apply_fn(
            params_dict, logits_t, ctx=contexts, cfl=conflicts,
            rngs=rngs_dict, **kwargs)
        # compute loss
        diff = logitsB - logitsA
        loss = mse_loss(output, diff)
        loss = jnp.where(batch["marker"], loss, jnp.zeros_like(loss))
        count = jnp.sum(batch["marker"])
        loss = jnp.where(count > 0, jnp.sum(loss)/count, 0)
        # collect metrics
        metrics = OrderedDict({"loss": loss*count, "count": count})
        metrics = jax.lax.psum(metrics, axis_name="batch")
        return metrics

    def step_sample(state, batch, config):
        ema_params, ema_cparams = state.ema_params
        context = f_normalize(batch["contexts"]) if config.context else None
        conflicts = batch["conflicts"] if config.conflict else None
        _supplesB = batch["supplesB"] if config.supple else None
        _supplesA = batch["supplesA"] if config.supple else None
        supplesB = normalize(_supplesB) if config.supple else None
        supplesA = normalize(_supplesA) if config.supple else None

        def apply(x_n, t_n, ctx, cfl):
            params_dict = dict(params=ema_params)
            rngs_dict = dict(dropout=state.rng)
            if state.batch_stats is not None:
                params_dict["batch_stats"] = state.batch_stats
            output = state.apply_fn(
                params_dict, x=x_n, t=t_n, ctx=ctx, cfl=cfl,
                rngs=rngs_dict, training=False)
            return output
        _logitsB = batch["images"]  # the current mode
        _logitsA = batch["labels"]  # mixture of other modes
        labels = batch["cls_labels"]
        logitsB = normalize(_logitsB)
        if config.supple:
            logitsB = jnp.concatenate([logitsB, supplesB], axis=-1)
        if config.conflict:
            logitsB = jnp.tile(logitsB, [2, 1])
            context = jnp.tile(context, [2, 1])
            cfl0 = jnp.zeros_like(labels)
            cfl1 = jnp.ones_like(labels)
            conflicts = jnp.concatenate([cfl0, cfl1], axis=0)
        if config.time_ensemble:
            raise NotImplementedError()
            # begin = time.time()
            # f_gen_all = state.sample2(
            #     state.rng, apply, logitsB, context, conflicts, dsb_stats)
            # sec = time.time()-begin
            # if config.supple:
            #     f_gen_all, a_gen_all = f_gen_all[...,
            #                                      :x_dim//2], f_gen_all[..., x_dim//2:]
            #     a_gen_all = jax.vmap(unnormalize)(a_gen_all)
            #     a_gen = a_gen_all[-1]
            # f_gen_all = jax.vmap(unnormalize)(f_gen_all)
            # f_gen = f_gen_all[-1]
        elif config.sample_ensemble > 1:
            begin = time.time()
            repeats = config.sample_ensemble
            expand = (1,)*len(logitsB.shape[1:])
            n_logitsB = jnp.tile(logitsB, reps=[repeats, *expand])
            n_context = jnp.tile(
                context, reps=[repeats, *((1,)*len(context.shape[1:]))]) if config.context else context
            n_conflicts = jnp.tile(
                conflicts, reps=[repeats, *((1,)*len(conflicts.shape[1:]))]) if config.conflict else conflicts
            f_gen_all = state.sample(
                apply, n_logitsB, n_context, n_conflicts, config.eps, config.T)
            sec = time.time()-begin
            f_gen_all = jnp.reshape(
                f_gen_all, (repeats, -1, *logitsB.shape[1:]))
            if config.supple:
                f_gen_all, a_gen_all = f_gen_all[...,
                                                 :x_dim//2], f_gen_all[..., x_dim//2:]
                a_gen_all = jax.vmap(unnormalize)(a_gen_all)
                a_gen = a_gen_all.mean(0)
            f_gen_all = jax.vmap(unnormalize)(f_gen_all)
            f_gen = f_gen_all.mean(0)
        else:
            begin = time.time()
            f_gen = state.sample(
                apply, logitsB, context, conflicts)
            sec = time.time()-begin
            if config.supple:
                f_gen, a_gen = f_gen[..., :x_dim//2], f_gen[..., x_dim//2:]
                a_gen = unnormalize(a_gen)
            f_gen = unnormalize(f_gen)
        f_real = _logitsA
        f_init = _logitsB
        a_real = _supplesA
        a_init = _supplesB
        if config.conflict:
            # product of expert
            f_gen = f_gen.reshape(2, -1, *f_gen.shape[1:]).mean(0)
            # TODO: f_gen_all for time_ensemble

        f_all = jnp.stack([f_init, f_gen, f_real], axis=-1)

        (
            (ens_acc, ens_nll),
            (
                (b_acc, b_nll), (a_acc, a_nll)
            )
        ) = ensemble_accuracy(
            [f_init, f_gen], labels, batch["marker"], ["B", "C"], ema_cparams)
        (
            ens_t2acc, (b_t2acc, a_t2acc)
        ) = ensemble_top2accuracy(
            [f_init, f_gen], labels, batch["marker"], ["B", "C"], ema_cparams)
        (
            ens_t5acc, (b_t5acc, a_t5acc)
        ) = ensemble_topNaccuracy(
            5, [f_init, f_gen], labels, batch["marker"], ["B", "C"], ema_cparams)
        (
            ens_tfpnacc, (b_tfpnacc, a_tfpnacc)
        ) = ensemble_tfpn_accuracy([f_init, f_gen], labels, batch["marker"],
                                   dict(tp=batch["tp"], fp=batch["fp"],
                                        fn=batch["fn"], tn=batch["tn"]),
                                   ["B", "C"], ema_cparams)
        if config.time_ensemble:
            ((tens_acc, tens_nll), _) = ensemble_accuracy(
                [f_init, f_init, f_init, f_gen_all[-3],
                    f_gen_all[-2], f_gen_all[-1]],
                labels, batch["marker"], [
                    "B", "B", "B", "C", "C", "C"], ema_cparams
            )
        count = jnp.sum(batch["marker"])
        kld = kl_divergence(
            f_real, f_gen, batch["marker"], "A", "C", ema_cparams)
        rkld = kl_divergence(
            f_gen, f_real, batch["marker"], "C", "A", ema_cparams)
        skld = kl_divergence(
            f_gen, f_init, batch["marker"], "C", "B", ema_cparams)
        rskld = kl_divergence(
            f_init, f_gen, batch["marker"], "B", "C", ema_cparams)
        if config.time_ensemble:
            tens_kld = ensemble_kld(
                [f_real], [f_gen_all[-3], f_gen_all[-2], f_gen_all[-1]],
                batch["marker"], "A", "C", ema_cparams)
            tens_rkld = ensemble_kld(
                [f_gen_all[-3], f_gen_all[-2], f_gen_all[-1]], [f_real],
                batch["marker"], "C", "A", ema_cparams)

        metrics = OrderedDict({
            "acc": a_acc, "nll": a_nll,
            "ens_acc": ens_acc, "ens_nll": ens_nll,
            "t2acc": a_t2acc, "ens_t2acc": ens_t2acc,
            "t5acc": a_t5acc, "ens_t5acc": ens_t5acc,
            "count": count,
            "kld": kld, "rkld": rkld,
            "skld": skld, "rskld": rskld,
            "sec": sec*count,
            "TPacc": a_tfpnacc["tp"],
            "FPacc": a_tfpnacc["fp"],
            "FNacc": a_tfpnacc["fn"],
            "TNacc": a_tfpnacc["tn"]
        })
        if config.time_ensemble:
            metrics["tens_acc"] = tens_acc
            metrics["tens_nll"] = tens_nll
            metrics["tens_kld"] = tens_kld
            metrics["tens_rkld"] = tens_rkld
        metrics = jax.lax.psum(metrics, axis_name='batch')

        C = get_logits(f_gen, "C", ema_cparams)
        A = get_logits(f_real, "A", ema_cparams)
        B = get_logits(f_init, "B", ema_cparams)
        aC = get_logits(a_gen, "C", ema_cparams) if config.supple else None
        aA = get_logits(a_real, "A", ema_cparams) if config.supple else None
        aB = get_logits(a_init, "B", ema_cparams) if config.supple else None
        supple_example = (aC, aA, aB, labels) if config.supple else None

        return f_all, metrics, (A, B, C, labels), supple_example

    def step_acc_ref(state, batch):
        _logitsB = batch["images"]  # the current mode
        _logitsA = batch["labels"]  # mixture of other modes
        labels = batch["cls_labels"]
        f_real = _logitsA
        f_init = _logitsB
        (
            (ens_acc, ens_nll),
            (
                (b_acc, b_nll), (a_acc, a_nll)
            )
        ) = ensemble_accuracy(
            [f_init, f_real], labels, batch["marker"], ["B", "A"])
        (
            ens_t2acc, (b_t2acc, a_t2acc)
        ) = ensemble_top2accuracy(
            [f_init, f_real], labels, batch["marker"], ["B", "A"])
        (
            ens_t5acc, (b_t5acc, a_t5acc)
        ) = ensemble_topNaccuracy(
            5, [f_init, f_real], labels, batch["marker"], ["B", "A"]
        )
        (
            ens_tfpnacc, (b_tfpnacc, a_tfpnacc)
        ) = ensemble_tfpn_accuracy(
            [f_init, f_real], labels, batch["marker"],
            dict(tp=batch["tp"], fp=batch["fp"],
                 fn=batch["fn"], tn=batch["tn"]),
            ["B", "A"]
        )
        rkld = kl_divergence(f_init, f_real, batch["marker"], "B", "A")
        kld = kl_divergence(f_real, f_init, batch["marker"], "A", "B")
        metrics = OrderedDict({
            "acc_ref": a_acc, "nll_ref": a_nll,
            "acc_from": b_acc, "nll_from": b_nll,
            "t2acc_ref": a_t2acc, "t2acc_from": b_t2acc,
            "t5acc_ref": a_t5acc, "t5acc_from": b_t5acc,
            "ens_acc_ref": ens_acc, "ens_nll_ref": ens_nll,
            "ens_t2acc_ref": ens_t2acc,
            "ens_t5acc_ref": ens_t5acc,
            "kld_ref": kld, "rkld_ref": rkld,
            "TPacc_ref": a_tfpnacc["tp"],
            "FPacc_ref": a_tfpnacc["fp"],
            "FNacc_ref": a_tfpnacc["fn"],
            "TNacc_ref": a_tfpnacc["tn"],
        })
        metrics = jax.lax.psum(metrics, axis_name='batch')
        return metrics

    def save_samples(state, batch, epoch_idx, train):
        if (epoch_idx+1) % 50 == 0 and os.environ.get("DEBUG") != True:
            z_all, _ = p_step_sample(state, batch)
            image_array = z_all[0][0]
            image_array = jax.nn.sigmoid(image_array)
            image_array = np.array(image_array)
            image_array = pixelize(image_array)
            image = Image.fromarray(image_array)
            image.save(train_image_name if train else valid_image_name)

    def save_advcls(examples, sexamples, name, epoch_idx):
        A, B, C, labels = examples
        aA, aB, aC, _ = sexamples
        assert len(labels.shape) == 2
        A = A.reshape(-1, *A.shape[2:])
        B = B.reshape(-1, *B.shape[2:])
        C = C.reshape(-1, *C.shape[2:])
        labels = labels.reshape(-1, *labels.shape[2:])
        aA = aA.reshape(-1, *aA.shape[2:])
        aB = aB.reshape(-1, *aB.shape[2:])
        aC = aC.reshape(-1, *aC.shape[2:])

        def top5(arr):
            top = jnp.argmax(arr, axis=-1)
            mask = common_utils.onehot(top, arr.shape[-1])
            arr = -mask*1e10+(1-mask)*arr
            mask = jnp.asarray(mask, dtype=bool)
            for i in range(1, 5):
                top = jnp.argmax(arr, axis=-1)
                mask2 = common_utils.onehot(top, arr.shape[-1])
                arr = -mask2*1e10+(1-mask2)*arr
                mask = jnp.logical_or(mask, mask2)
            return mask
        # index of batch that was correct in B but not in A
        Apred = jax.nn.log_softmax(A, axis=-1)
        Acorrect = evaluate_acc(
            Apred, labels, log_input=True, reduction="none")
        Bpred = jax.nn.log_softmax(B, axis=-1)
        Bcorrect = evaluate_acc(
            Bpred, labels, log_input=True, reduction="none")
        Acorrect = jnp.asarray(Acorrect, dtype=bool)
        Bcorrect = jnp.asarray(Bcorrect, dtype=bool)
        save_maskA = jnp.logical_and(Acorrect, ~Bcorrect)
        save_maskB = jnp.logical_and(~Acorrect, Bcorrect)
        save_maskAB = jnp.logical_and(Acorrect, Bcorrect)
        save_mask_ = jnp.logical_and(~Acorrect, ~Bcorrect)

        aApred = jax.nn.log_softmax(aA, axis=-1)
        aBpred = jax.nn.log_softmax(aB, axis=-1)
        aCpred = jax.nn.log_softmax(aC, axis=-1)

        def adv_plot(save_mask, dis):
            if jnp.any(save_mask):
                save_labelA = top5(Apred[save_mask][0])
                save_labelB = top5(Bpred[save_mask][0])
                save_label = jnp.logical_or(save_labelA, save_labelB)
                Cpred = jax.nn.log_softmax(C, axis=-1)
                _A = jnp.exp(Apred[save_mask][0][save_label])
                _B = jnp.exp(Bpred[save_mask][0][save_label])
                _C = jnp.exp(Cpred[save_mask][0][save_label])
                _labels = labels[save_mask][0]
                _aA = jnp.exp(aApred[save_mask][0][save_label])
                _aB = jnp.exp(aBpred[save_mask][0][save_label])
                _aC = jnp.exp(aCpred[save_mask][0][save_label])
                cls_plot(_A, _B, _C, save_label, _labels,
                         name+dis, epoch_idx)
                cls_plot(_aA, _aB, _aC, save_label, _labels,
                         name+"-div"+dis, epoch_idx)
        adv_plot(save_maskA, "B")
        adv_plot(save_maskB, "A")
        adv_plot(save_maskAB, "AB")
        adv_plot(save_mask_, "_")

    def save_cls(A, B, C, labels, name, epoch_idx):
        assert len(labels.shape) == 2
        A = A.reshape(-1, *A.shape[2:])
        B = B.reshape(-1, *B.shape[2:])
        C = C.reshape(-1, *C.shape[2:])
        labels = labels.reshape(-1, *labels.shape[2:])

        def top5(arr):
            top = jnp.argmax(arr, axis=-1)
            mask = common_utils.onehot(top, arr.shape[-1])
            arr = -mask*1e10+(1-mask)*arr
            mask = jnp.asarray(mask, dtype=bool)
            for i in range(1, 5):
                top = jnp.argmax(arr, axis=-1)
                mask2 = common_utils.onehot(top, arr.shape[-1])
                arr = -mask2*1e10+(1-mask2)*arr
                mask = jnp.logical_or(mask, mask2)
            return mask
        # index of batch that was correct in B but not in A
        Apred = jax.nn.log_softmax(A, axis=-1)
        Acorrect = evaluate_acc(
            Apred, labels, log_input=True, reduction="none")
        Bpred = jax.nn.log_softmax(B, axis=-1)
        Bcorrect = evaluate_acc(
            Bpred, labels, log_input=True, reduction="none")
        Acorrect = jnp.asarray(Acorrect, dtype=bool)
        Bcorrect = jnp.asarray(Bcorrect, dtype=bool)
        save_maskA = jnp.logical_and(Acorrect, ~Bcorrect)
        save_maskB = jnp.logical_and(~Acorrect, Bcorrect)
        save_maskAB = jnp.logical_and(Acorrect, Bcorrect)
        save_mask_ = jnp.logical_and(~Acorrect, ~Bcorrect)

        def plot(save_mask, dis):
            if jnp.any(save_mask):
                save_labelA = top5(Apred[save_mask][0])
                save_labelB = top5(Bpred[save_mask][0])
                save_label = jnp.logical_or(save_labelA, save_labelB)
                Cpred = jax.nn.log_softmax(C, axis=-1)
                _A = jnp.exp(Apred[save_mask][0][save_label])
                _B = jnp.exp(Bpred[save_mask][0][save_label])
                _C = jnp.exp(Cpred[save_mask][0][save_label])
                _labels = labels[save_mask][0]
                cls_plot(_A, _B, _C, save_label, _labels, name+dis, epoch_idx)
        plot(save_maskA, "B")
        plot(save_maskB, "A")
        plot(save_maskAB, "AB")
        plot(save_mask_, "_")

    cross_replica_mean = jax.pmap(lambda x: jax.lax.pmean(x, 'x'), 'x')
    p_step_train = jax.pmap(
        partial(step_train, config=config), axis_name="batch")
    p_step_valid = jax.pmap(step_valid, axis_name="batch")
    p_step_sample = jax.pmap(
        partial(step_sample, config=config), axis_name="batch")
    p_step_acc_ref = jax.pmap(step_acc_ref, axis_name="batch")
    state = jax_utils.replicate(state)
    best_loss = float("inf")
    best_acc = 0
    image_name = datetime.datetime.now().strftime('%y%m%d%H%M%S%f')
    feature_type = "last" if config.features_dir in feature_dir_list else "logit"
    train_image_name = f"images/{config.version}train{feature_type}{image_name}.png"
    valid_image_name = f"images/{config.version}valid{feature_type}{image_name}.png"
    trn_summary = dict()
    val_summary = dict()
    tst_summary = dict()
    bank = FeatureBank(config.n_classes,
                       maxlen=config.optim_bs,
                       disable=True if config.mixup_alpha <= 0 else False,
                       gamma=config.mixup_gamma)

    if config.save_cls2:
        test_loader = dataloaders["tst_featureloader"](rng=None)
        test_loader = jax_utils.prefetch_to_device(test_loader, size=2)
        save_examples = None
        for batch_idx, batch in enumerate(test_loader, start=1):
            rng = jax.random.fold_in(rng, batch_idx)
            state = state.replace(rng=jax_utils.replicate(rng))
            _, _, examples, _ = p_step_sample(state, batch)
            A, B, C, labels = examples
            time_stamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            path = getattr(
                config, "path", f"{config.features_dir}{time_stamp}")
            epoch_idx = ckpt.get("step") or ""
            save_cls(A, B, C, labels, path, epoch_idx)

    wandb.init(
        project="dsb-bnn",
        config=vars(config),
        mode="disabled" if config.nowandb else "online"
    )
    wandb.define_metric("val/loss", summary="min")
    wandb.define_metric("val/acc", summary="max")
    wandb.define_metric("val/ens_acc", summary="max")
    wandb.define_metric("val/ens_nll", summary="min")
    wandb.run.summary["params"] = sum(
        x.size for x in jax.tree_util.tree_leaves(variables["params"]))
    wl = WandbLogger()
    if config.nowandb:
        time_stamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        path = f"{config.features_dir}{time_stamp}"
    else:
        path = wandb.run.name
    config.path = path
    save_path = os.path.abspath(f"./score_checkpoints/{path}")

    for epoch_idx, _ in enumerate(tqdm(range(total_epochs)), start=1):
        rng = jax.random.fold_in(rng, epoch_idx)
        data_rng, rng = jax.random.split(rng)

        # -----------------------------------------------------------------
        # training
        # -----------------------------------------------------------------
        train_metrics = []
        if epoch_idx <= config.optim_ne:
            train_loader = dataloaders["featureloader"](rng=data_rng)
        else:
            train_loader = dataloaders["val_featureloader"](rng=None)
        train_loader = jax_utils.prefetch_to_device(train_loader, size=2)
        for batch_idx, batch in enumerate(train_loader, start=1):
            rng = jax.random.fold_in(rng, batch_idx)
            batch = bank.mixup_inclass(rng, batch, alpha=config.mixup_alpha)
            if config.permaug:
                batch = bank.perm_aug(rng, batch)
            state = state.replace(rng=jax_utils.replicate(rng))
            state, metrics = p_step_train(state, batch)

            if epoch_idx == 1:
                acc_ref_metrics = p_step_acc_ref(state, batch)
                metrics.update(acc_ref_metrics)
            if config.show:
                if batch_idx == dataloaders["trn_steps_per_epoch"]:
                    save_samples(state, batch, epoch_idx, train=True)
            else:
                if (epoch_idx+1) % 50 == 0:
                    _, acc_metrics, _, _ = p_step_sample(state, batch)
                    metrics.update(acc_metrics)

            train_metrics.append(metrics)

        train_metrics = common_utils.get_metrics(train_metrics)
        trn_summarized = {f'trn/{k}': v for k,
                          v in jax.tree_util.tree_map(lambda e: e.sum(), train_metrics).items()}
        for k, v in trn_summarized.items():
            if "TP" in k:
                trn_summarized[k] /= dataloaders["train_tfpn"]["tp"]
            elif "FP" in k:
                trn_summarized[k] /= dataloaders["train_tfpn"]["fp"]
            elif "FN" in k:
                trn_summarized[k] /= dataloaders["train_tfpn"]["fn"]
            elif "TN" in k:
                trn_summarized[k] /= dataloaders["train_tfpn"]["tn"]
            elif "count" not in k and "lr" not in k:
                trn_summarized[k] /= trn_summarized["trn/count"]
        del trn_summarized["trn/count"]
        trn_summary.update(trn_summarized)
        wl.log(trn_summary)

        if state.batch_stats is not None:
            state = state.replace(
                batch_stats=cross_replica_mean(state.batch_stats))

        # -----------------------------------------------------------------
        # validation
        # -----------------------------------------------------------------
        valid_metrics = []
        valid_loader = dataloaders["val_featureloader"](rng=None)
        valid_loader = jax_utils.prefetch_to_device(valid_loader, size=2)
        for batch_idx, batch in enumerate(valid_loader, start=1):
            rng = jax.random.fold_in(rng, batch_idx)
            state = state.replace(rng=jax_utils.replicate(rng))
            metrics = p_step_valid(state, batch)

            if epoch_idx == 1:
                acc_ref_metrics = p_step_acc_ref(state, batch)
                metrics.update(acc_ref_metrics)
            if config.show:
                if batch_idx == dataloaders["val_steps_per_epoch"]:
                    save_samples(state, batch, epoch_idx, train=False)
            else:
                if (epoch_idx+1) % 1 == 0:
                    _, acc_metrics, _, _ = p_step_sample(state, batch)
                    metrics.update(acc_metrics)
                    assert jnp.all(metrics["count"] == acc_metrics["count"])

            valid_metrics.append(metrics)

        valid_metrics = common_utils.get_metrics(valid_metrics)
        val_summarized = {f'val/{k}': v for k,
                          v in jax.tree_util.tree_map(lambda e: e.sum(), valid_metrics).items()}
        for k, v in val_summarized.items():
            if "TP" in k:
                val_summarized[k] /= dataloaders["valid_tfpn"]["tp"]
            elif "FP" in k:
                val_summarized[k] /= dataloaders["valid_tfpn"]["fp"]
            elif "FN" in k:
                val_summarized[k] /= dataloaders["valid_tfpn"]["fn"]
            elif "TN" in k:
                val_summarized[k] /= dataloaders["valid_tfpn"]["tn"]
            elif "count" not in k and "lr" not in k:
                val_summarized[k] /= val_summarized["val/count"]
        del val_summarized["val/count"]
        val_summary.update(val_summarized)
        wl.log(val_summary)

        # -----------------------------------------------------------------
        # testing
        # -----------------------------------------------------------------
        criteria = val_summarized["val/ens_acc"] if not config.am051101 else val_summarized["val/acc"]
        if best_acc < criteria or config.testall:
            test_metrics = []
            test_loader = dataloaders["tst_featureloader"](rng=None)
            test_loader = jax_utils.prefetch_to_device(test_loader, size=2)
            save_examples = None
            for batch_idx, batch in enumerate(test_loader, start=1):
                rng = jax.random.fold_in(rng, batch_idx)
                state = state.replace(rng=jax_utils.replicate(rng))
                _, metrics, examples, sexamples = p_step_sample(state, batch)
                if best_acc == 0:
                    acc_ref_metrics = p_step_acc_ref(state, batch)
                    metrics.update(acc_ref_metrics)
                if batch_idx == 1:
                    save_examples = examples
                if config.save_cls:
                    if not config.supple:
                        A, B, C, labels = examples
                        save_cls(A, B, C, labels, path, epoch_idx)
                    else:
                        save_advcls(examples, sexamples, path, epoch_idx)
                test_metrics.append(metrics)
            test_metrics = common_utils.get_metrics(test_metrics)
            tst_summarized = {
                f'tst/{k}': v for k, v in jax.tree_util.tree_map(lambda e: e.sum(), test_metrics).items()}
            for k, v in tst_summarized.items():
                if "TP" in k:
                    tst_summarized[k] /= dataloaders["test_tfpn"]["tp"]
                elif "FP" in k:
                    tst_summarized[k] /= dataloaders["test_tfpn"]["fp"]
                elif "FN" in k:
                    tst_summarized[k] /= dataloaders["test_tfpn"]["fn"]
                elif "TN" in k:
                    tst_summarized[k] /= dataloaders["test_tfpn"]["tn"]
                elif "count" not in k and "lr" not in k:
                    tst_summarized[k] /= tst_summarized["tst/count"]
            del tst_summarized["tst/count"]
            tst_summary.update(tst_summarized)
            wl.log(tst_summary)
            if best_acc < criteria:
                best_acc = criteria

            if config.save:
                save_state = jax_utils.unreplicate(state)
                ckpt = dict(model=save_state,
                            config=vars(config),
                            best_acc=best_acc,
                            examples=dict(
                                A=save_examples[0],
                                B=save_examples[1],
                                C=save_examples[2],
                                labels=save_examples[3],
                            ))
                orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
                checkpoints.save_checkpoint(ckpt_dir=save_path,
                                            target=ckpt,
                                            step=epoch_idx,
                                            overwrite=True,
                                            orbax_checkpointer=orbax_checkpointer)
        wl.flush()

        # wait until computations are done
        jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
        if jnp.isnan(trn_summarized['trn/loss']):
            print("NaN detected")
            break

    wandb.finish()


def main():

    TIME_STAMP = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    parser = defaults.default_argument_parser()

    # ---------------------------------------------------------------------------------------
    # optimizer
    # ---------------------------------------------------------------------------------------
    parser.add_argument('--optim_ne', default=350, type=int,
                        help='the number of training epochs (default: 200)')
    parser.add_argument('--optim_lr', default=2e-4, type=float,
                        help='base learning rate (default: 1e-4)')
    parser.add_argument('--optim_momentum', default=0.9, type=float,
                        help='momentum coefficient (default: 0.9)')
    parser.add_argument('--optim_weight_decay', default=0.1, type=float,
                        help='weight decay coefficient (default: 0.0001)')
    # ---------------------------------------------------------------------------------------
    # training
    # ---------------------------------------------------------------------------------------
    parser.add_argument('--save', action="store_true")
    parser.add_argument('--seed', default=2023, type=int,
                        help='random seed for training (default: None)')
    parser.add_argument('--precision', default='fp32', type=str,
                        choices=['fp16', 'fp32'])
    parser.add_argument(
        "--features_dir", default="features100_fixed", type=str)
    parser.add_argument("--context", action="store_true")
    parser.add_argument(
        "--contexts_dir", default="features100_last_fixed", type=str)
    parser.add_argument("--supple", action="store_true")
    parser.add_argument("--gamma", default=0., type=float)
    parser.add_argument("--beta", default=0., type=float)
    parser.add_argument("--start_rglr", default=0, type=int)
    parser.add_argument("--rglr_list", default="rkld", type=str)
    parser.add_argument("--rglr_coef", default="1", type=str)
    parser.add_argument("--nowandb", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--get_stats", action="store_true")
    parser.add_argument("--n_Amodes", default=1, type=int)
    parser.add_argument("--n_samples_each_mode", default=1, type=int)
    parser.add_argument("--take_valid", action="store_true")
    parser.add_argument("--ema_decay", default=0.999, type=float)
    parser.add_argument("--mixup_alpha", default=0., type=float)
    parser.add_argument("--mixup_gamma", default=1., type=float)
    parser.add_argument("--permaug", action="store_true")
    parser.add_argument("--trainset_only", action="store_true")
    parser.add_argument("--validset_only", action="store_true")
    parser.add_argument("--finetune_ne", default=0, type=int)
    parser.add_argument("--mse_power", default=2, type=int)
    parser.add_argument("--p2_weight", action="store_true")
    parser.add_argument("--p2_gamma", action="store_true")
    parser.add_argument("--p2_gammainv", action="store_true")
    parser.add_argument("--eps", default=1e-3, type=float)
    # ---------------------------------------------------------------------------------------
    # experiemnts
    # ---------------------------------------------------------------------------------------
    # score matching until 10 epochs(early stop), freeze the score-net, and finetune with KLD and linear layer
    parser.add_argument("--am050911", action="store_true")
    # am050911 with p2_gamma or p2_gammainv
    parser.add_argument("--pm050902", action="store_true")
    # look for where the test accuracy saturates # pm050903
    parser.add_argument("--testall", action="store_true")
    # put features of A as contexts to check the algorithm works correctly (cheating)
    parser.add_argument("--pm050905", action="store_true")
    # save classification plots
    parser.add_argument("--save_cls", action="store_true")
    # learn ensembled logits
    parser.add_argument("--pm051010", action="store_true")
    # learn averaged logits
    parser.add_argument("--am051101", action="store_true")
    # get instances whose inference results are different between A and B
    parser.add_argument("--get_conflict_rate", action="store_true")
    # enhance loss where A inference conflicts with B
    parser.add_argument("--pm051101", action="store_true")
    # utilize conflict prediction information as score net conditionals
    parser.add_argument("--conflict", action="store_true")
    # save classification plots from restored model
    parser.add_argument("--save_cls2", action="store_true")
    # enhance loss where A inference does not conflicts with B (opposite to pm051101)
    parser.add_argument("--pm053105", action="store_true")
    # dsb between two modes close to each other
    parser.add_argument("--pm060704", default=0, type=int)
    # ---------------------------------------------------------------------------------------
    # diffusion
    # ---------------------------------------------------------------------------------------
    parser.add_argument("--T", default=5, type=int)
    parser.add_argument("--beta1", default=1e-4, type=float)
    parser.add_argument("--beta2", default=0.02, type=float)
    parser.add_argument("--time_ensemble", action="store_true")
    parser.add_argument("--sample_ensemble", default=0, type=int)
    parser.add_argument("--determ_eps", action="store_true")
    parser.add_argument("--maxmask", action="store_true")
    # ---------------------------------------------------------------------------------------
    # networks
    # ---------------------------------------------------------------------------------------
    parser.add_argument("--n_feat", default=256, type=int)
    parser.add_argument("--version", default="v1.0", type=str)
    parser.add_argument("--droprate", default=0.2, type=float)
    parser.add_argument("--stop_grad", action="store_true")
    parser.add_argument("--corrector", default=1, type=int)
    # UNetModel only
    parser.add_argument("--learn_sigma", action="store_true")
    parser.add_argument("--num_res_blocks", default=1, type=int)
    parser.add_argument("--num_heads", default=1, type=int)
    parser.add_argument("--use_scale_shift_norm", action="store_true")
    parser.add_argument("--resblock_updown", action="store_true")
    parser.add_argument("--use_new_attention_order", action="store_true")

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

    settings = __import__(f"{args.features_dir}.settings", fromlist=[""])
    for k, v in vars(settings).items():
        if "__" in k:
            continue
        if isinstance(v, int) or isinstance(v, float) or isinstance(v, str) or isinstance(v, bool):
            setattr(args, k, v)
    del settings

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
    main()
