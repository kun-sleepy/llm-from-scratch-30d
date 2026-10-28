import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(d_model))   # 可学习增益，初始为 1

    def forward(self, x):
        # x: (..., d_model) —— 对最后一维归一化
        ms = x.pow(2).mean(dim=-1, keepdim=True)      # 均方（mean of squares）
        return x * torch.rsqrt(ms + self.eps) * self.g # rsqrt = 1/√，一次 kernel 搞定



norm = RMSNorm(64)
x = torch.randn(2, 10, 64) * 7 + 3      # 故意加大尺度和偏移
y = norm(x)
# 归一化后每行的均方应 ≈ 1（因为 g=1）
assert torch.allclose(y.pow(2).mean(-1), torch.ones(2, 10), atol=1e-3)
# 关键性质验证：RMSNorm 对输入缩放不变（放大 100 倍输出几乎不变）
assert torch.allclose(norm(100*x), norm(x), atol=1e-2)




    