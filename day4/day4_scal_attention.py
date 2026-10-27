import torch
import torch.nn.functional as F

# def scaled_dot_product_attention(Q, K, V, mask=None):
#     """
#     Q, K, V: (batch, n_heads, seq_len, d_k)
#     mask:    可广播到 (batch, n_heads, seq_len, seq_len) 的布尔张量,True=允许看
#     """
#     d_k = Q.size(-1)
#     scores = Q @ K.transpose(-2, -1) / d_k ** 0.5   # (B, H, L, L) —— 预习题1的答案
#     if mask is not None:
#         # 预习题3答案：softmax 之前！把禁止位置设为 -inf
#         # 为什么 -inf 有效：softmax 里 exp(-inf)=0，权重精确归零
#         scores = scores.masked_fill(~mask, float('-inf'))
#     attn = torch.softmax(scores, dim=-1)             # 每行和为 1 —— 预习题2中"谁归一化"
#     return attn @ V, attn                            # 返回输出和权重矩阵（权重留着画图）



def scaled_dot_product_attention(Q, K, V, mask=None):

    d_k=Q.size(-1)
    scores=Q @ K.transpose(-2,-1) / d_k**0.5

    if mask is not None: 
        scores=scores.masked_fill(~mask,float('-inf'))

    attn=torch.softmax(scores,dim=-1)

    return attn @ V , attn


B, H, L, D = 2, 4, 8, 16
Q = torch.randn(B, H, L, D); K = torch.randn(B, H, L, D); V = torch.randn(B, H, L, D)

my_out, _ = scaled_dot_product_attention(Q, K, V)
ref_out = F.scaled_dot_product_attention(Q, K, V)   # PyTorch 官方
assert torch.allclose(my_out, ref_out, atol=1e-6) , "手写 Attention 输出与官方标准不一致！"















