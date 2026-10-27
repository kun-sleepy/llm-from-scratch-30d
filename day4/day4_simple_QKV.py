
import torch
X = torch.eye(2)
scores = X @ X.T / (2 ** 0.5)
attn = torch.softmax(scores, dim=-1)
out = attn @ X
print(attn, out)


# dk and softmax
d_k = 512
q = torch.randn(10000, d_k); k = torch.randn(10000, d_k)
s = (q * k).sum(-1)
print(s.std().item())   # 应 ≈ √512 ≈ 22.6 —— 与推导对拍

# 观察 softmax 饱和：大输入 → 输出近似 one-hot → 梯度≈0
z = torch.tensor([23.0, -5.0, 10.0])
p = torch.softmax(z, dim=-1); print(p)          # ≈ [1, 0, 0]
p.sum().backward  # 思考：softmax 输出饱和时 dz 的梯度是什么量级？

