# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright 2020 Ross Wightman
# Modified model creation / weight loading / state_dict helpers

import logging
import os
import math
from collections import OrderedDict
from copy import deepcopy
from typing import Callable

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from timesformer.models.features import FeatureListNet, FeatureDictNet, FeatureHookNet
from timesformer.models.conv2d_same import Conv2dSame
from timesformer.models.linear import Linear


_logger = logging.getLogger(__name__)

def _pca_project_rows(weight, out_dim):
    if out_dim >= weight.shape[0]:
        return weight, None
    w = weight.float()
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    u_r = u[:, :out_dim]
    w_r = (s[:out_dim].unsqueeze(1) * vh[:out_dim, :])
    return w_r.to(weight.dtype), u_r.to(weight.dtype)


def _convert_vit_qkv_to_dna(state_dict, model, q_neg_scale=1e-4, v_neg_scale=-1e-4, use_pca=True):
    model_keys = set(model.state_dict().keys())
    wants_qqkvv = any(".qqkvv." in k for k in model_keys)
    wants_qv = any(".qv." in k for k in model_keys) and any(".k_proj." in k for k in model_keys)
    has_qqkvv = any(".qqkvv." in k for k in state_dict)
    has_qv = any(".qv." in k for k in state_dict)
    has_qkv = any(".qkv." in k for k in state_dict)

    if not has_qkv:
        return state_dict

    if wants_qqkvv and not has_qqkvv:
        remove_keys = []
        for key in list(state_dict.keys()):
            if not key.endswith("qkv.weight"):
                continue
            new_key = key.replace("qkv.weight", "qqkvv.weight")
            if new_key not in model_keys:
                continue
            qkv = state_dict[key]
            if qkv.ndim != 2 or qkv.shape[0] % 3 != 0:
                continue
            c = qkv.shape[0] // 3
            q, k, v = qkv[:c], qkv[c:2 * c], qkv[2 * c:3 * c]
            q_neg = q * q_neg_scale
            v_neg = v * v_neg_scale
            state_dict[new_key] = torch.cat([q, q_neg, k, v, v_neg], dim=0)
            remove_keys.append(key)

            bias_key = key.replace("weight", "bias")
            if bias_key in state_dict:
                bias = state_dict[bias_key]
                if bias.numel() == 3 * c:
                    q_b, k_b, v_b = bias[:c], bias[c:2 * c], bias[2 * c:3 * c]
                    q_neg_b = q_b * q_neg_scale
                    v_neg_b = v_b * v_neg_scale
                    state_dict[new_key.replace("weight", "bias")] = torch.cat(
                        [q_b, q_neg_b, k_b, v_b, v_neg_b], dim=0
                    )
                    remove_keys.append(bias_key)

        for key in remove_keys:
            state_dict.pop(key, None)
        return state_dict

    if wants_qv and not has_qv:
        remove_keys = []
        for key in list(state_dict.keys()):
            if not key.endswith("qkv.weight"):
                continue
            qv_key = key.replace("qkv.weight", "qv.weight")
            k_key = key.replace("qkv.weight", "k_proj.weight")
            if qv_key not in model_keys or k_key not in model_keys:
                continue
            qkv = state_dict[key]
            if qkv.ndim != 2 or qkv.shape[0] % 3 != 0:
                continue
            c = qkv.shape[0] // 3
            if c % 2 != 0:
                continue
            half = c // 2
            q, k, v = qkv[:c], qkv[c:2 * c], qkv[2 * c:3 * c]
            if use_pca:
                q_pos, u_q = _pca_project_rows(q, half)
                k_proj, u_k = _pca_project_rows(k, half)
                v_pos, u_v = _pca_project_rows(v, half)
            else:
                q_pos, k_proj, v_pos = q[:half], k[:half], v[:half]
                u_q = u_k = u_v = None

            q_neg = q_pos * q_neg_scale
            v_neg = v_pos * v_neg_scale
            state_dict[qv_key] = torch.cat([q_pos, q_neg, v_pos, v_neg], dim=0)
            state_dict[k_key] = k_proj
            remove_keys.append(key)

            bias_key = key.replace("weight", "bias")
            if bias_key in state_dict:
                bias = state_dict[bias_key]
                if bias.numel() == 3 * c:
                    q_b, k_b, v_b = bias[:c], bias[c:2 * c], bias[2 * c:3 * c]
                    if use_pca:
                        q_pos_b = u_q.T @ q_b if u_q is not None else q_b[:half]
                        k_b_r = u_k.T @ k_b if u_k is not None else k_b[:half]
                        v_pos_b = u_v.T @ v_b if u_v is not None else v_b[:half]
                    else:
                        q_pos_b = q_b[:half]
                        k_b_r = k_b[:half]
                        v_pos_b = v_b[:half]
                    q_neg_b = q_pos_b * q_neg_scale
                    v_neg_b = v_pos_b * v_neg_scale
                    state_dict[qv_key.replace("weight", "bias")] = torch.cat(
                        [q_pos_b, q_neg_b, v_pos_b, v_neg_b], dim=0
                    )
                    state_dict[k_key.replace("weight", "bias")] = k_b_r
                    remove_keys.append(bias_key)

        for key in remove_keys:
            state_dict.pop(key, None)
        return state_dict

    return state_dict


def _map_dna_positive_to_standard_attention(state_dict, model):
    """For standard attention modules, ensure qkv weights exist by extracting the
    positive branch from dna-style weights (e.g., qqkvv) if needed."""
    for name, module in model.named_modules():
        if not name:
            continue
        if not hasattr(module, "qkv"):
            continue
        if not isinstance(getattr(module, "qkv", None), nn.Linear):
            continue
        # Standard attention has qkv but no dna-specific params.
        if hasattr(module, "q_neg") or hasattr(module, "v_neg") or hasattr(module, "qqkvv") or hasattr(module, "qv"):
            continue

        qkv_w_key = f"{name}.qkv.weight"
        qkv_b_key = f"{name}.qkv.bias"
        if qkv_w_key in state_dict:
            continue

        qqkvv_w_key = f"{name}.qqkvv.weight"
        qqkvv_b_key = f"{name}.qqkvv.bias"
        if qqkvv_w_key not in state_dict:
            continue

        w = state_dict[qqkvv_w_key]
        if w.ndim != 2 or w.shape[0] % 5 != 0:
            _logger.warning(
                "Cannot map %s to %s: unexpected shape %s",
                qqkvv_w_key,
                qkv_w_key,
                tuple(w.shape),
            )
            continue

        c = w.shape[0] // 5
        q_pos = w[:c]
        k = w[2 * c:3 * c]
        v_pos = w[3 * c:4 * c]
        state_dict[qkv_w_key] = torch.cat([q_pos, k, v_pos], dim=0)

        if qqkvv_b_key in state_dict and qkv_b_key not in state_dict:
            b = state_dict[qqkvv_b_key]
            if b.ndim == 1 and b.shape[0] == 5 * c:
                q_b = b[:c]
                k_b = b[2 * c:3 * c]
                v_b = b[3 * c:4 * c]
                state_dict[qkv_b_key] = torch.cat([q_b, k_b, v_b], dim=0)
            else:
                _logger.warning(
                    "Cannot map %s to %s: unexpected shape %s",
                    qqkvv_b_key,
                    qkv_b_key,
                    tuple(b.shape),
                )

    return state_dict

def load_state_dict(checkpoint_path, use_ema=False):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict_key = 'state_dict'
        if isinstance(checkpoint, dict):
            if use_ema and 'state_dict_ema' in checkpoint:
                state_dict_key = 'state_dict_ema'
        if state_dict_key and state_dict_key in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `module.` prefix
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        elif 'model_state' in checkpoint:
            state_dict_key = 'model_state'
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `model.` prefix
                name = k[6:] if k.startswith('model') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            state_dict = checkpoint
        _logger.info("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        return state_dict
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_checkpoint(model, checkpoint_path, use_ema=False, strict=True):
    state_dict = load_state_dict(checkpoint_path, use_ema)
    model.load_state_dict(state_dict, strict=strict)


def resume_checkpoint(model, checkpoint_path, optimizer=None, loss_scaler=None, log_info=True):
    resume_epoch = None
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            if log_info:
                _logger.info('Restoring model state from checkpoint...')
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

            if optimizer is not None and 'optimizer' in checkpoint:
                if log_info:
                    _logger.info('Restoring optimizer state from checkpoint...')
                optimizer.load_state_dict(checkpoint['optimizer'])

            if loss_scaler is not None and loss_scaler.state_dict_key in checkpoint:
                if log_info:
                    _logger.info('Restoring AMP loss scaler state from checkpoint...')
                loss_scaler.load_state_dict(checkpoint[loss_scaler.state_dict_key])

            if 'epoch' in checkpoint:
                resume_epoch = checkpoint['epoch']
                if 'version' in checkpoint and checkpoint['version'] > 1:
                    resume_epoch += 1  # start at the next epoch, old checkpoints incremented before save

            if log_info:
                _logger.info("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
        else:
            model.load_state_dict(checkpoint)
            if log_info:
                _logger.info("Loaded checkpoint '{}'".format(checkpoint_path))
        return resume_epoch
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()

def check_and_load_weights(model, state_dict_to_load, strict=False):
    print("--- Starting weight check ---")

    initial_weights = OrderedDict()
    for name, param in model.state_dict().items():
        initial_weights[name] = param.clone()

    print(f"Loading new state dictionary (strict={strict})...")
    model.load_state_dict(state_dict_to_load, strict=strict)
    print("Loading complete.")

    changed_layers = []
    unchanged_layers = []
    new_layers = []

    final_weights = model.state_dict()

    for name, final_param in final_weights.items():
        if name not in initial_weights:
            new_layers.append(name)
            continue

        initial_param = initial_weights[name]

        if not torch.equal(initial_param, final_param):
            changed_layers.append(name)
        else:
            unchanged_layers.append(name)

    print("\n--- Weight Comparison Report ---")
    if changed_layers:
        print(f"Found {len(changed_layers)} layers with UPDATED weights:")
        for name in changed_layers:
            print(f"  - {name}")
    else:
        print("\nNo layers had their weights updated.")

    if unchanged_layers:
        print(f"\nFound {len(unchanged_layers)} layers with UNCHANGED weights.")
        for name in unchanged_layers:
            print(f"  - {name}")        
    else:
        print("\nNo layers remained unchanged.")
    
    if new_layers:
        print(f"\nFound {len(new_layers)} NEW layers (not in the initial model state):")
        for name in new_layers:
            print(f"  - {name}")

    print("\n--- Check complete ---\n")
    
    return model


def load_pretrained(model, cfg=None, num_classes=1000, in_chans=3, filter_fn=None, img_size=224, num_frames=8, num_patches=196, attention_type='divided_space_time', pretrained_model="", strict=True, dna_convert=False, dna_q_neg_scale=1e-4, dna_v_neg_scale=-1e-4, dna_use_pca=True, dna_positive_only=False):
    if cfg is None:
        cfg = getattr(model, 'default_cfg')
    if cfg is None or 'url' not in cfg or not cfg['url']:
        _logger.warning("Pretrained model URL is invalid, using random initialization.")
        return

    if len(pretrained_model) == 0:
        _logger.info(
            "==== PRETRAINED SOURCE: TIMM URL (%s) | dna_convert=%s ====",
            cfg.get("url", "unknown"),
            dna_convert,
        )
        state_dict = model_zoo.load_url(cfg['url'], progress=False, map_location='cpu')
    elif pretrained_model.endswith('.pyth'):
        _logger.info(
            "==== PRETRAINED SOURCE: LOCAL CHECKPOINT (%s) | dna_convert=False (Forced) ====",
            pretrained_model,
        )
        state_dict = load_state_dict(pretrained_model)
        dna_convert = False
    else:
        _logger.info(
            "==== PRETRAINED SOURCE: LOCAL CHECKPOINT (%s) | dna_convert=%s ====",
            pretrained_model,
            dna_convert,
        )
        try:
            state_dict = load_state_dict(pretrained_model)['model']
        except Exception:
            state_dict = load_state_dict(pretrained_model)

    
    
    if filter_fn is not None and not pretrained_model.endswith('.pyth'):
        state_dict = filter_fn(state_dict)

    if dna_convert:
        state_dict = _convert_vit_qkv_to_dna(
            state_dict,
            model,
            q_neg_scale=dna_q_neg_scale,
            v_neg_scale=dna_v_neg_scale,
            use_pca=dna_use_pca,
        )

    if in_chans == 1:
        conv1_name = cfg['first_conv']
        _logger.info('Converting first conv (%s) pretrained weights from 3 to 1 channel' % conv1_name)
        conv1_weight = state_dict[conv1_name + '.weight']
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        O, I, J, K = conv1_weight.shape
        if I > 3:
            assert conv1_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv1_weight = conv1_weight.reshape(O, I // 3, 3, J, K)
            conv1_weight = conv1_weight.sum(dim=2, keepdim=False)
        else:
            conv1_weight = conv1_weight.sum(dim=1, keepdim=True)
        conv1_weight = conv1_weight.to(conv1_type)
        state_dict[conv1_name + '.weight'] = conv1_weight
    elif in_chans != 3:
        conv1_name = cfg['first_conv']
        conv1_weight = state_dict[conv1_name + '.weight']
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        O, I, J, K = conv1_weight.shape
        if I != 3:
            _logger.warning('Deleting first conv (%s) from pretrained weights.' % conv1_name)
            del state_dict[conv1_name + '.weight']
            strict = False
        else:
            _logger.info('Repeating first conv (%s) weights in channel dim.' % conv1_name)
            repeat = int(math.ceil(in_chans / 3))
            conv1_weight = conv1_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
            conv1_weight *= (3 / float(in_chans))
            conv1_weight = conv1_weight.to(conv1_type)
            state_dict[conv1_name + '.weight'] = conv1_weight


    classifier_name = cfg['classifier']
    if num_classes == 1000 and cfg['num_classes'] == 1001:
        # special case for imagenet trained models with extra background class in pretrained weights
        classifier_weight = state_dict[classifier_name + '.weight']
        state_dict[classifier_name + '.weight'] = classifier_weight[1:]
        classifier_bias = state_dict[classifier_name + '.bias']
        state_dict[classifier_name + '.bias'] = classifier_bias[1:]
    elif num_classes != state_dict[classifier_name + '.weight'].size(0):
        #print('Removing the last fully connected layer due to dimensions mismatch ('+str(num_classes)+ ' != '+str(state_dict[classifier_name + '.weight'].size(0))+').', flush=True)
        # completely discard fully connected for all other differences between pretrained and created model
        del state_dict[classifier_name + '.weight']
        del state_dict[classifier_name + '.bias']
        strict = False


    ## Resizing the positional embeddings in case they don't match
    if num_patches + 1 != state_dict['pos_embed'].size(1):
        pos_embed = state_dict['pos_embed']
        cls_pos_embed = pos_embed[0,0,:].unsqueeze(0).unsqueeze(1)
        other_pos_embed = pos_embed[0,1:,:].unsqueeze(0).transpose(1, 2)
        new_pos_embed = F.interpolate(other_pos_embed, size=(num_patches), mode='nearest')
        new_pos_embed = new_pos_embed.transpose(1, 2)
        new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
        state_dict['pos_embed'] = new_pos_embed

    ## Resizing time embeddings in case they don't match
    if 'time_embed' in state_dict and num_frames != state_dict['time_embed'].size(1):
        time_embed = state_dict['time_embed'].transpose(1, 2)
        new_time_embed = F.interpolate(time_embed, size=(num_frames), mode='nearest')
        state_dict['time_embed'] = new_time_embed.transpose(1, 2)

    ## Initializing temporal attention
    if attention_type == 'divided_space_time':
        new_state_dict = state_dict.copy()
        for key in state_dict:
            if 'blocks' in key and 'attn' in key:
                new_key = key.replace('attn','temporal_attn')
                if not new_key in state_dict:
                   new_state_dict[new_key] = state_dict[key]
                else:
                   new_state_dict[new_key] = state_dict[new_key]
            if 'blocks' in key and 'norm1' in key:
                new_key = key.replace('norm1','temporal_norm1')
                if not new_key in state_dict:
                   new_state_dict[new_key] = state_dict[key]
                else:
                   new_state_dict[new_key] = state_dict[new_key]
        state_dict = new_state_dict

    if dna_positive_only:
        state_dict = _map_dna_positive_to_standard_attention(state_dict, model)

    ## Loading the weights
    # model.load_state_dict(state_dict, strict=False)
    model = check_and_load_weights(model, state_dict, strict=False)
    print(model)


def extract_layer(model, layer):
    layer = layer.split('.')
    module = model
    if hasattr(model, 'module') and layer[0] != 'module':
        module = model.module
    if not hasattr(model, 'module') and layer[0] == 'module':
        layer = layer[1:]
    for l in layer:
        if hasattr(module, l):
            if not l.isdigit():
                module = getattr(module, l)
            else:
                module = module[int(l)]
        else:
            return module
    return module


def set_layer(model, layer, val):
    layer = layer.split('.')
    module = model
    if hasattr(model, 'module') and layer[0] != 'module':
        module = model.module
    lst_index = 0
    module2 = module
    for l in layer:
        if hasattr(module2, l):
            if not l.isdigit():
                module2 = getattr(module2, l)
            else:
                module2 = module2[int(l)]
            lst_index += 1
    lst_index -= 1
    for l in layer[:lst_index]:
        if not l.isdigit():
            module = getattr(module, l)
        else:
            module = module[int(l)]
    l = layer[lst_index]
    setattr(module, l, val)


def adapt_model_from_string(parent_module, model_string):
    separator = '***'
    state_dict = {}
    lst_shape = model_string.split(separator)
    for k in lst_shape:
        k = k.split(':')
        key = k[0]
        shape = k[1][1:-1].split(',')
        if shape[0] != '':
            state_dict[key] = [int(i) for i in shape]

    new_module = deepcopy(parent_module)
    for n, m in parent_module.named_modules():
        old_module = extract_layer(parent_module, n)
        if isinstance(old_module, nn.Conv2d) or isinstance(old_module, Conv2dSame):
            if isinstance(old_module, Conv2dSame):
                conv = Conv2dSame
            else:
                conv = nn.Conv2d
            s = state_dict[n + '.weight']
            in_channels = s[1]
            out_channels = s[0]
            g = 1
            if old_module.groups > 1:
                in_channels = out_channels
                g = in_channels
            new_conv = conv(
                in_channels=in_channels, out_channels=out_channels, kernel_size=old_module.kernel_size,
                bias=old_module.bias is not None, padding=old_module.padding, dilation=old_module.dilation,
                groups=g, stride=old_module.stride)
            set_layer(new_module, n, new_conv)
        if isinstance(old_module, nn.BatchNorm2d):
            new_bn = nn.BatchNorm2d(
                num_features=state_dict[n + '.weight'][0], eps=old_module.eps, momentum=old_module.momentum,
                affine=old_module.affine, track_running_stats=True)
            set_layer(new_module, n, new_bn)
        if isinstance(old_module, nn.Linear):
            num_features = state_dict[n + '.weight'][1]
            new_fc = Linear(
                in_features=num_features, out_features=old_module.out_features, bias=old_module.bias is not None)
            set_layer(new_module, n, new_fc)
            if hasattr(new_module, 'num_features'):
                new_module.num_features = num_features
    new_module.eval()
    parent_module.eval()

    return new_module


def adapt_model_from_file(parent_module, model_variant):
    adapt_file = os.path.join(os.path.dirname(__file__), 'pruned', model_variant + '.txt')
    with open(adapt_file, 'r') as f:
        return adapt_model_from_string(parent_module, f.read().strip())


def default_cfg_for_features(default_cfg):
    default_cfg = deepcopy(default_cfg)
    # remove default pretrained cfg fields that don't have much relevance for feature backbone
    to_remove = ('num_classes', 'crop_pct', 'classifier')  # add default final pool size?
    for tr in to_remove:
        default_cfg.pop(tr, None)
    return default_cfg


def build_model_with_cfg(
        model_cls: Callable,
        variant: str,
        pretrained: bool,
        default_cfg: dict,
        model_cfg: dict = None,
        feature_cfg: dict = None,
        pretrained_strict: bool = True,
        pretrained_filter_fn: Callable = None,
        **kwargs):
    pruned = kwargs.pop('pruned', False)
    features = False
    feature_cfg = feature_cfg or {}

    if kwargs.pop('features_only', False):
        features = True
        feature_cfg.setdefault('out_indices', (0, 1, 2, 3, 4))
        if 'out_indices' in kwargs:
            feature_cfg['out_indices'] = kwargs.pop('out_indices')

    model = model_cls(**kwargs) if model_cfg is None else model_cls(cfg=model_cfg, **kwargs)
    model.default_cfg = deepcopy(default_cfg)

    if pruned:
        model = adapt_model_from_file(model, variant)

    # for classification models, check class attr, then kwargs, then default to 1k, otherwise 0 for feats
    num_classes_pretrained = 0 if features else getattr(model, 'num_classes', kwargs.get('num_classes', 1000))
    if pretrained:
        load_pretrained(
            model,
            num_classes=num_classes_pretrained, in_chans=kwargs.get('in_chans', 3),
            filter_fn=pretrained_filter_fn, strict=pretrained_strict)

    if features:
        feature_cls = FeatureListNet
        if 'feature_cls' in feature_cfg:
            feature_cls = feature_cfg.pop('feature_cls')
            if isinstance(feature_cls, str):
                feature_cls = feature_cls.lower()
                if 'hook' in feature_cls:
                    feature_cls = FeatureHookNet
                else:
                    assert False, f'Unknown feature class {feature_cls}'
        model = feature_cls(model, **feature_cfg)
        model.default_cfg = default_cfg_for_features(default_cfg)  # add back default_cfg

    return model
