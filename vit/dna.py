import torch
import torch.nn as nn
import torch.nn.functional as F


class DenoisingAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        learnable_attn=None,
        layer_idx=None,
    ):
        del learnable_attn
        super().__init__()
        head_dim = dim // num_heads
        self.num_heads = num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.layer_idx = layer_idx
        self.C = dim

        self.qqkvv = nn.Linear(dim, dim * 5, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.neg_scalers = nn.Parameter(torch.ones(num_heads) * 0.5)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        qqkvv_weight_key = prefix + "qqkvv.weight"
        legacy_weight_keys = [
            prefix + "qkv.weight",
            prefix + "q_neg.weight",
            prefix + "v_neg.weight",
        ]
        if qqkvv_weight_key not in state_dict and all(key in state_dict for key in legacy_weight_keys):
            state_dict[qqkvv_weight_key] = torch.cat(
                [state_dict.pop(key) for key in legacy_weight_keys],
                dim=0,
            )

        qqkvv_bias_key = prefix + "qqkvv.bias"
        legacy_bias_keys = [
            prefix + "qkv.bias",
            prefix + "q_neg.bias",
            prefix + "v_neg.bias",
        ]
        if qqkvv_bias_key not in state_dict and all(key in state_dict for key in legacy_bias_keys):
            state_dict[qqkvv_bias_key] = torch.cat(
                [state_dict.pop(key) for key in legacy_bias_keys],
                dim=0,
            )

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x, epoch=None):
        del epoch
        batch_size, tokens, channels = x.shape
        qqkvv = self.qqkvv(x).reshape(batch_size, tokens, 5, self.num_heads, channels // self.num_heads)
        qqkvv = qqkvv.permute(2, 0, 3, 1, 4)
        q_pos, q_neg, k, v_pos, v_neg = qqkvv[0], qqkvv[1], qqkvv[2], qqkvv[3], qqkvv[4]

        q_pos = q_pos * self.scale
        q_neg = q_neg * self.scale

        attn_pos = q_pos @ k.transpose(-2, -1)
        attn_neg = q_neg @ k.transpose(-2, -1)

        pos_attn = self.attn_drop(F.softmax(attn_pos, dim=-1))
        neg_attn = self.attn_drop(F.softmax(-attn_neg, dim=-1))
        neg_attn = self.neg_scalers.view(1, -1, 1, 1) * neg_attn

        x = (pos_attn @ v_pos - neg_attn @ v_neg).transpose(1, 2).reshape(batch_size, tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, None, None


class DenoisingAttentionSharedValues(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        learnable_attn=None,
        layer_idx=None,
    ):
        del learnable_attn
        super().__init__()
        head_dim = dim // num_heads
        self.num_heads = num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.layer_idx = layer_idx
        self.C = dim

        self.qqkv = nn.Linear(dim, dim * 4, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.neg_scalers = nn.Parameter(torch.ones(num_heads) * 0.5)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        qqkv_weight_key = prefix + "qqkv.weight"
        legacy_weight_keys = [
            prefix + "qkv.weight",
            prefix + "q_neg.weight",
        ]
        if qqkv_weight_key not in state_dict and all(key in state_dict for key in legacy_weight_keys):
            state_dict[qqkv_weight_key] = torch.cat(
                [state_dict.pop(key) for key in legacy_weight_keys],
                dim=0,
            )

        qqkv_bias_key = prefix + "qqkv.bias"
        legacy_bias_keys = [
            prefix + "qkv.bias",
            prefix + "q_neg.bias",
        ]
        if qqkv_bias_key not in state_dict and all(key in state_dict for key in legacy_bias_keys):
            state_dict[qqkv_bias_key] = torch.cat(
                [state_dict.pop(key) for key in legacy_bias_keys],
                dim=0,
            )

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x, epoch=None):
        del epoch
        batch_size, tokens, channels = x.shape
        qqkv = self.qqkv(x).reshape(batch_size, tokens, 4, self.num_heads, channels // self.num_heads)
        qqkv = qqkv.permute(2, 0, 3, 1, 4)
        q_pos, q_neg, k, v = qqkv[0], qqkv[1], qqkv[2], qqkv[3]

        q_pos = q_pos * self.scale
        q_neg = q_neg * self.scale

        attn_pos = q_pos @ k.transpose(-2, -1)
        attn_neg = q_neg @ k.transpose(-2, -1)

        pos_attn = self.attn_drop(F.softmax(attn_pos, dim=-1))
        neg_attn = self.attn_drop(F.softmax(-attn_neg, dim=-1))
        neg_attn = self.neg_scalers.view(1, -1, 1, 1) * neg_attn

        final_attn = pos_attn - neg_attn

        x = (final_attn @ v).transpose(1, 2).reshape(batch_size, tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, None, None
