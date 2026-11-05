import sys; sys.path.insert(0, '.')
import torch
from day5_sim_GPT import GPT

# 推荐配置（Colab T4 友好）：约 20M 参数
CFG = dict(vocab_size=50257, d_model=384, n_heads=6, d_ff=1024,
           n_layers=6, max_len=128)
model = GPT(**CFG)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"参数量: {n_params/1e6:.1f}M")
# ✅ 手算对拍：Embedding 50257×384≈19.3M + 6层×[4×384²+3×384×1024]≈6×1.77M≈10.6M ≈ 30M
# （Embedding 占大头是正常的——weight tying 已帮你省掉一份）

# 显存自查（预习题5）：Adam 状态 = 2×参数 float32 = 2×30M×4B = 240MB
# 参数+梯度+状态 ≈ 4×30M×4B = 480MB，batch 激活另算——T4 15GB 宽裕
