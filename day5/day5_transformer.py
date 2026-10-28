import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def scaled_dot_product_attention(Q, K, V, mask=None):

    d_k=Q.size(-1)
    # scores=Q @ K.transpose(-2,-1) / d_k**0.5

    scores=torch.einsum('bhld,bhkd->bhlk',Q,K)/ d_k**0.5

    if mask is not None: 
        scores=scores.masked_fill(~mask,float('-inf'))

    attn=torch.softmax(scores,dim=-1)

    return attn @ V , attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        # 四个投影：Q K V 合并/分开均可，O 输出投影 —— 预习题4的参数量就在这四个矩阵里
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, L, D = x.shape
        # (B, L, D) → 投影 → 拆头 (B, L, H, d_k) → 转置 (B, H, L, d_k)
        Q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        out, attn = scaled_dot_product_attention(Q, K, V, mask)
        # 拼头：(B, H, L, d_k) → (B, L, D)，注意 transpose 后内存不连续要 contiguous()
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.W_o(out), attn

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



class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(d_model))   # 可学习增益，初始为 1

    def forward(self, x):
        # x: (..., d_model) —— 对最后一维归一化
        ms = x.pow(2).mean(dim=-1, keepdim=True)      # 均方（mean of squares）
        return x * torch.rsqrt(ms + self.eps) * self.g # rsqrt = 1/√，一次 kernel 搞定


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_len):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_qkv = nn.Linear(d_model, 3*d_model, bias=False)  # QKV 合并：一次 GEMM 更快
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        cos, sin = precompute_rope_cache(max_len, self.d_k)
        self.register_buffer('cos', cos, persistent=False)   # buffer：不进参数、随 .to(device) 走
        self.register_buffer('sin', sin, persistent=False)
        self.register_buffer('mask', torch.tril(torch.ones(max_len, max_len, dtype=torch.bool)), persistent=False)

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.W_qkv(x).view(B, L, 3, self.n_heads, self.d_k)
        Q, K, V = (qkv[:, :, i].transpose(1, 2) for i in range(3))   # 各 (B,H,L,d_k)
        Q = apply_rope(Q, self.cos[:L], self.sin[:L])
        K = apply_rope(K, self.cos[:L], self.sin[:L])     # 注意：只旋转 q 和 k，V 不转！
        # scores = Q @ K.transpose(-2,-1) / self.d_k**0.5
        scores = torch.einsum('BHLD,BHKD->BHLK',Q,K)/ self.d_k**0.5
        scores = scores.masked_fill(~self.mask[:L,:L], float('-inf'))
        out = torch.softmax(scores, dim=-1) @ V
        # out = out.transpose(1, 2).contiguous().view(B, L, D)
        out=torch.einsum('BHLD->BLHD',out).reshape(B, L, D)
        return self.W_o(out)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # 注意：图中写的是 x W_1^T，PyTorch nn.Linear 内部自带了转置，所以直接定义即可
        self.w1 = nn.Linear(d_model, d_ff, bias=False) # 主路 W1
        self.v  = nn.Linear(d_model, d_ff, bias=False) # 门控路 V
        self.w2 = nn.Linear(d_ff, d_model, bias=False) # 降维 W2

    def forward(self, x):
        # 1. 主路经过 SiLU
        main_path = F.silu(self.w1(x))
        # 2. 门控路不激活
        gate_path = self.v(x)
        # 3. 逐元素相乘 (⊙) 并降维
        return self.w2(main_path * gate_path)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, max_len):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn  = CausalSelfAttention(d_model, n_heads, max_len)
        self.norm2 = RMSNorm(d_model)
        self.ffn   = SwiGLU(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))    # pre-norm 残差 1
        x = x + self.ffn(self.norm2(x))     # pre-norm 残差 2
        return x


blk = TransformerBlock(d_model=512, n_heads=8, d_ff=2048, max_len=128)
x = torch.randn(2, 64, 512) # B L D
out = blk(x)
assert out.shape == x.shape
# 参数量对拍单元 3 的手算
n = sum(p.numel() for p in blk.parameters())
expected = 3*512*512 + 512*512 + 3*512*2048 + 2*512   # QKV合并+O + FFN + 两个norm的g
assert n == expected, (n, expected)
# 梯度流测试：输出对输入求导不 NaN（pre-norm 的意义）
x.requires_grad_(True); blk(x).sum().backward()
assert not torch.isnan(x.grad).any()

