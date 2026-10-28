import torch
import torch.nn as nn
import torch.nn.functional as F

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


ffn = SwiGLU(512, 2048)


# numel 是 Number of Elements
n = sum(p.numel() for p in ffn.parameters())
# 手算：3 × 512 × 2048 = 3,145,728（无 bias）
assert n == 3 * 512 * 2048
# 整个 Block 参数量 = attention(4×512² = 1,048,576) + FFN(3,145,728) + 2×RMSNorm(1024)
# ≈ 4,195,328 ≈ 4.2M —— 手算一遍，与稍后组装的 Block 对拍

    
ffn = SwiGLU(2, 8)
for p in ffn.parameters():
    print(p)

