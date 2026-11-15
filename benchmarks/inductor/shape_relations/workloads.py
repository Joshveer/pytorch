"""Team reproduction functions, kept unchanged for the benchmark.

Source: CyrusD123/inductor-dynamic-fusion at
275f764ee84812a8db53ff93e915577f9b97b2bc, repro_split_heads.py and
repro_dim_equality.py, and repro_xsa_issue.py. Only the original unchecked
equality workload is used.
"""

import torch
import torch.nn.functional as F


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


def xsa_unsqueeze(output, xv):
    head_dim = xv.shape[-1]
    n_kv = xv.shape[2]
    n_heads = output.shape[2]
    torch._check(output.shape[3] == head_dim)
    torch._check(n_heads % n_kv == 0)
    n_rep = n_heads // n_kv

    vn = F.normalize(xv, dim=-1, eps=1e-6).unsqueeze(3)
    y = output.view(*output.shape[:2], n_kv, n_rep, head_dim)
    ps = (y * vn).sum(dim=-1, keepdim=True)
    cos_sim = ps.squeeze(-1).abs()
    proj_frac = cos_sim / y.norm(dim=-1).clamp(min=1e-6)
    result = (y - ps * vn).view_as(output)
    return result, cos_sim, proj_frac


def xsa_expand(output, xv):
    B, T, Hq, D = output.shape
    Hkv = xv.shape[2]
    torch._check(xv.shape[3] == D)
    torch._check(Hq % Hkv == 0)
    gs = Hq // Hkv

    v_exp = xv[:, :, :, None, :].expand(B, T, Hkv, gs, D).reshape(B, T, Hq, D)
    vn = F.normalize(v_exp, p=2, dim=-1, eps=1e-6)
    coeff = (output * vn).sum(dim=-1, keepdim=True)
    projected = output - coeff * vn
    cg = coeff.squeeze(-1).view(B, T, Hkv, gs)
    cos_sim = cg.abs()
    yg = output.view(B, T, Hkv, gs, D)
    proj_frac = cos_sim / yg.norm(dim=-1).clamp(min=1e-6)
    return projected, cos_sim, proj_frac
