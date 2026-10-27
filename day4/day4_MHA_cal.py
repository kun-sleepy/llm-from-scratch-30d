import torch
import torch.nn as nn

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


mha = MultiHeadAttention(d_model=64, n_heads=4)
x = torch.randn(2, 10, 64)

# 1. 形状检查
out, attn = mha(x, mask=torch.tril(torch.ones(10, 10, dtype=torch.bool)))
assert out.shape == (2, 10, 64) and attn.shape == (2, 4, 10, 10)

# 2. 参数量手算对拍（预习题4）：4 个方阵 = 4×(64×64+64) = 16,640
n_params = sum(p.numel() for p in mha.parameters())
assert n_params == 4 * (64*64 + 64), n_params

# 3. 把 Day1 的 einsum 用进来：将 scores 计算改写为 einsum 版本并对拍
scores_ref = Q_check = None  # 自行用 torch.einsum('bhld,bhkd->bhlk', Q, K) 重写并验证






    