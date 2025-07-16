# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention as Attention_, Mlp



class FlashAttention(Attention_):
    """Multi-head Flash Attention block with qk norm."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        qk_norm=False,
        **block_kwargs,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
        """
        super().__init__(dim, num_heads=num_heads, qkv_bias=qkv_bias, **block_kwargs)

        if qk_norm:
            self.q_norm = nn.LayerNorm(dim)
            self.k_norm = nn.LayerNorm(dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x, mask=None, HW=None, block_id=None, **kwargs):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, C)
        q, k, v = qkv.unbind(2)
        dtype = q.dtype

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.reshape(B, N, self.num_heads, C // self.num_heads).to(dtype)
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).to(dtype)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).to(dtype)

        use_fp32_attention = getattr(self, "fp32_attention", False)  # necessary for NAN loss
        if use_fp32_attention:
            q, k, v = q.float(), k.float(), v.float()

        attn_bias = None
        if mask is not None:
            attn_bias = torch.zeros([B * self.num_heads, q.shape[1], k.shape[1]], dtype=q.dtype, device=q.device)
            attn_bias.masked_fill_(mask.squeeze(1).repeat(self.num_heads, 1, 1) == 0, float("-inf"))

        if _xformers_available:
            x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)
        else:
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            if mask is not None and mask.ndim == 2:
                mask = (1 - mask.to(x.dtype)) * -10000.0
                mask = mask[:, None, None].repeat(1, self.num_heads, 1, 1)
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
            x = x.transpose(1, 2)

        x = x.view(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if torch.get_autocast_gpu_dtype() == torch.float16:
            x = x.clip(-65504, 65504)

        return x


#https://github.com/NVlabs/Sana/blob/70459f414474c10c509e8b58f3f9442738f85577/diffusion/model/nets/sana_blocks.py#L453
#################################################################################
#   AMP attention with fp32 softmax to fix loss NaN problem during training     #
#################################################################################
class Attention(Attention_):
    def forward(self, x, HW=None, **kwargs):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # B,N,3,H,C -> B,H,N,C
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        use_fp32_attention = getattr(self, "fp32_attention", False)
        if use_fp32_attention:
            q, k = q.float(), k.float()

        with torch.cuda.amp.autocast(enabled=not use_fp32_attention):
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    


try:
    import flash_attn

    if hasattr(flash_attn, "__version__") and int(flash_attn.__version__[0]) == 2:
        from flash_attn.flash_attn_interface import flash_attn_kvpacked_func
        from flash_attn.modules.mha import FlashSelfAttention
    else:
        from flash_attn.flash_attn_interface import flash_attn_unpadded_kvpacked_func
        from flash_attn.modules.mha import FlashSelfAttention
except Exception as e:
    print(f"flash_attn import failed: {e}")

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)




class FlashSelfMHAModified(nn.Module):
    """
    self-attention with flashattention
    """

    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=True,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        device=None,
        dtype=None,
        norm_layer=nn.LayerNorm,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.dim // num_heads
        assert (
            self.head_dim % 8 == 0 and self.head_dim <= 128
        ), "Only support head_dim <= 128 and divisible by 8"

        self.Wqkv = nn.Linear(dim, 3 * dim, bias=qkv_bias, **factory_kwargs)
        # TODO: eps should be 1 / 65530 if using fp16
        self.q_norm = (
            norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6)
            if qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6)
            if qk_norm
            else nn.Identity()
        )
        self.inner_attn = FlashSelfAttention(attention_dropout=attn_drop)
        self.out_proj = nn.Linear(dim, dim, bias=qkv_bias, **factory_kwargs)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        x,
    ):
        """
        Parameters
        ----------
        x: torch.Tensor
            (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        """
        b, s, d = x.shape

        qkv = self.Wqkv(x)
        qkv = qkv.view(b, s, 3, self.num_heads, self.head_dim)  # [b, s, 3, h, d]
        q, k, v = qkv.unbind(dim=2)  # [b, s, h, d]
        q = self.q_norm(q).half()  # [b, s, h, d]
        k = self.k_norm(k).half()

        qkv = torch.stack([q, k, v], dim=2)  # [b, s, 3, h, d]
        context = self.inner_attn(qkv)
        out = self.out_proj(context.view(b, s, d))
        out = self.proj_drop(out)

        return out


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256,max_period=10000):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period

    #@staticmethod
    def timestep_embedding(self,t, dim):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0,att_type="raw", **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if  att_type=="flash":
            self.attn = FlashAttention(
                hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True
            )
        elif att_type=="flex":
            raise NotImplementedError("Flex attention is not implemented")
            from torch.nn.attention.flex_attention import flex_attention
            self.attn = flex_attention
        elif att_type=="raw":
            self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        elif att_type=="fsdp":
            self.attn = Attention_(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        else:
            raise NotImplementedError(f"Attention type {att_type} not implemented")
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=8,
        out_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        #num_classes=1000,
        num_classes=-1,
        learn_sigma=False,
        use_dt_adaptor=False,
        use_shortcut=True,
        name="dit",
        att_type="fsdp",
    ):
        super().__init__()
        self.name = name
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = out_channels #in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_shortcut = use_shortcut

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size, max_period=10_000)# as t range from 0 to 1
        if use_shortcut:
            self.dt_embedder = TimestepEmbedder(hidden_size, max_period=10_000)# as dt range from 0 to 1
            if use_dt_adaptor:
                self.dt_adaptor = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size, bias=True),
                    nn.SiLU(),
                    nn.Linear(hidden_size, hidden_size, bias=True),
                    nn.SiLU(),
                    nn.Linear(hidden_size, hidden_size, bias=True),
                )
            else:
                self.dt_adaptor = nn.Identity()
        if num_classes > 0 and False:
            self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        else:
            self.y_embedder = None
        self.num_classes = num_classes
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,att_type=att_type) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        if self.y_embedder is not None:
            # Initialize label embedding table:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        if self.use_shortcut:
            nn.init.normal_(self.dt_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.dt_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    #@torch.compile(mode="max-autotune")
    def forward(self, x, t, dt, **kwargs):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        dt: (N,) tensor of diffusion timesteps
        """        
        if dt is not None and len(x.shape)==len(dt.shape):
            dt = dt.squeeze(-1).squeeze(-1).squeeze(-1)
        if len(x.shape)==len(t.shape):
            t = t.squeeze(-1).squeeze(-1).squeeze(-1)

        y = kwargs.get("y", None)
        assert y is not None and len(y.shape) == len(x.shape), f"y.shape: {y.shape}, x.shape: {x.shape}"
        x = torch.cat([x,y],dim=1)
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t) 
        c = t 
        

        if dt is not None:
            dt = self.dt_embedder(dt) 
            dt = self.dt_adaptor(dt)
            c = c + dt                              # (N, D)
        if self.y_embedder is not None and False:
             y = kwargs.get("y", None)
             y = self.y_embedder(y, self.training)    # (N, D)
             c = c +  y                             # (N, D)
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    def forward_without_cfg(self, x, t, dt,**kwargs):
       return self.forward(x, t, dt, **kwargs)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}

if __name__ == "__main__":
    model = DiT_XL_2(use_dt_adaptor=True)
    x = torch.randn(1, 4, 32, 32)
    t = torch.rand(1,)
    y = torch.randint(0, 1000, (1,))
    td = torch.rand(1,)
    print(model(x, t, td).shape)
