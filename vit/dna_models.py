from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import Mlp, PatchEmbed

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
        neg_attn = self.attn_drop(F.softmin(attn_neg, dim=-1))
        neg_attn = self.neg_scalers.view(1, -1, 1, 1) * neg_attn

        x = (pos_attn @ v_pos + neg_attn @ v_neg).transpose(1, 2).reshape(batch_size, tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, None, None


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        Attention_block=DenoisingAttention,
        Mlp_block=Mlp,
        layer_idx=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            layer_idx=layer_idx,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        y, _, _ = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DnA_Vision_Transformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        block_layers=Block,
        Patch_layer=PatchEmbed,
        act_layer=nn.GELU,
        Attention_block=DenoisingAttention,
        Mlp_block=Mlp,
        **kwargs,
    ):
        super().__init__()
        self.dropout_rate = drop_rate
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = Patch_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        dpr = [drop_path_rate for _ in range(depth)]
        self.blocks = nn.ModuleList(
            [
                block_layers(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=0.0,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[idx],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    Attention_block=Attention_block,
                    Mlp_block=Mlp_block,
                    layer_idx=idx,
                )
                for idx in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        batch_size = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = x + self.pos_embed
        x = torch.cat((cls_tokens, x), dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        if self.dropout_rate:
            x = F.dropout(x, p=float(self.dropout_rate), training=self.training)
        return self.head(x)


@register_model
def dna_vit_base_patch16_224(pretrained=False, img_size=224, pretrained_21k=False, **kwargs):
    del pretrained
    del pretrained_21k
    return DnA_Vision_Transformer(
        img_size=img_size,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=Block,
        Attention_block=DenoisingAttention,
        **kwargs,
    )