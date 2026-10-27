import torch


L = 8
# torch.tril  torch.triu  下三角 triangular + lower  
causal_mask = torch.tril(torch.ones(L, L, dtype=torch.bool))   # 下三角 True
print(causal_mask.int())

scores = torch.randn(L, L)

# masked_fill 更换true的地方
# & | ~ ^  与 或 非 异或
masked = scores.masked_fill(~causal_mask, float('-inf'))
attn = torch.softmax(masked, dim=-1)
print(attn.round(decimals=2))   # 上三角应全为 0，每行和为 1


def scaled_dot_product_attention(Q, K, V, mask=None):

    d_k=Q.size(-1)
    scores=Q @ K.transpose(-2,-1) / d_k**0.5

    if mask is not None: 
        scores=scores.masked_fill(~mask,float('-inf'))

    attn=torch.softmax(scores,dim=-1)

    return attn @ V , attn


# 构造输入 X，跑带因果 mask 的 attention，
# 然后只修改 X 的最后一个 token，再跑一次：
# 如果 mask 正确，前 L-1 个输出必须完全不变！
X = torch.randn(1, 1, 8, 16)
out1, _ = scaled_dot_product_attention(X, X, X, mask=causal_mask)
X2 = X.clone()
print(f"{X2[..., -1, :].size()}")
X2[..., -1, :] += 100     # 剧烈修改最后一个 token
print(f"{X[..., -1, :]}")
print(f"{X2[..., -1, :]}")
out2, _ = scaled_dot_product_attention(X2, X2, X2, mask=causal_mask)
assert torch.allclose(out1[..., :-1, :], out2[..., :-1, :])   # 前 7 个位置纹丝不动
print("无信息泄漏, mask 正确")




