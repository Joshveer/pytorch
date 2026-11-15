"""Team reproduction functions, kept unchanged for the benchmark.

Source: CyrusD123/inductor-dynamic-fusion at
275f764ee84812a8db53ff93e915577f9b97b2bc, repro_split_heads.py and
repro_dim_equality.py. Only the original unchecked equality workload is used.
"""

import torch


H = 8


def softmax_split(x):
    B, T, C = x.shape
    return torch.softmax(x, -1).view(B, T, H, C // H) * 2.0


def rmsnorm_split(x, w_head):
    """OLMo-2-style QK-norm: RMSNorm over all of C, then split into heads and
    apply a per-head-dim scale (a stand-in for RoPE / any pointwise)."""
    B, T, C = x.shape
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    return y.view(B, T, H, C // H) * w_head


def layernorm_split(x):
    B, T, C = x.shape
    y = torch.nn.functional.layer_norm(x, (C,))
    return y.view(B, T, H, C // H) * 2.0


def no_split(x):
    return torch.softmax(x, -1) * 2.0


def stats_from_other(x, y):
    return y * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
