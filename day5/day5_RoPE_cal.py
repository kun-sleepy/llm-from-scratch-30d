import torch
import torch.nn as nn


def precompute_rope_cache(seq_len, d_k, base=10000.0, device='cpu'):
    # theta_i = base^(-2i/d)，共 d_k/2 个频率
    inv_freq = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))   # (d_k/2,)
    t = torch.arange(seq_len).float()                                     # 位置 m
    freqs = torch.outer(t, inv_freq)          # (seq_len, d_k/2)：每个位置×每个平面的角度 m·θ_i
    cos = freqs.cos(); sin = freqs.sin()      # 预计算，推理时查表
    return cos.to(device), sin.to(device)

def apply_rope(x, cos, sin):
    # x: (B, H, L, d_k)；cos/sin: (L, d_k/2)
    x1, x2 = x[..., ::2], x[..., 1::2]        # 拆成偶/奇两路 = d/2 个二维平面
    # 二维旋转公式：[x1'; x2'] = [x1·cos − x2·sin, x1·sin + x2·cos]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    out = torch.stack([out1, out2], dim=-1).flatten(-2)   # 交错还原
    return out



d_k, L = 16, 32
cos, sin = precompute_rope_cache(L, d_k)
q0 = torch.randn(1, 1, 1, d_k); k0 = torch.randn(1, 1, 1, d_k)

def dot_at(m, n):   # q 放在位置 m，k 放在位置 n，算内积
    q_m = apply_rope(q0, cos[m:m+1], sin[m:m+1])
    k_n = apply_rope(k0, cos[n:n+1], sin[n:n+1])
    return (q_m @ k_n.transpose(-2,-1)).item()

# 核心断言：内积只依赖 (m−n)！
d1 = dot_at(10, 5); d2 = dot_at(20, 15)     # 相对距离都是 5
assert abs(d1 - d2) < 1e-5, (d1, d2)
print(f"位置(10,5)与(20,15)的内积相同：{d1:.4f} ≈ {d2:.4f} —— 相对位置性质 ✓")


