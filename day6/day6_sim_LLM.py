import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# ================= 1. 基础配置 =================
class Config:
    vocab_size = 1000      # 假设我们有一个包含1000个词的字符集
    block_size = 64        # 上下文长度（每次模型看64个词）
    n_embd = 128           # 模型特征维度
    batch_size = 16
    max_steps = 500
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

cfg = Config()

# ================= 2. 数据准备（错位采样） =================
# 模拟文本：随机生成0-999的整数，代表token ids
raw_data = np.random.randint(0, cfg.vocab_size, size=50000)

class TokenDataset:
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
        
    def sample_batch(self, batch_size):
        ix = torch.randint(0, len(self.data) - self.block_size - 1, (batch_size,))
        x = torch.stack([torch.from_numpy(self.data[i : i+self.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i+1 : i+self.block_size+1].astype(np.int64)) for i in ix])
        return x.to(cfg.device), y.to(cfg.device)

dataset = TokenDataset(raw_data, cfg.block_size)

# ================= 3. 极简模型定义 =================
# 这里用一个最简单的单层线性模型代替 Transformer，重点是展示输入输出维度
class DummyLLM(nn.Module):
    def __init__(self, vocab_size, n_embd):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size) # 输出维度必须等于字符集大小
        
    def forward(self, x):
        # x: (B, L)
        h = self.embed(x)             # (B, L, n_embd)
        logits = self.lm_head(h)      # (B, L, vocab_size)
        return logits

model = DummyLLM(cfg.vocab_size, cfg.n_embd).to(cfg.device)

# ================= 4. 学习率调度（Warmup + Cosine） =================
def get_lr(step, total_steps, max_lr=3e-4, min_lr=1e-5, warmup_steps=50):
    # 1. 预热阶段
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    # 2. 余弦退火阶段
    if step > total_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# ================= 5. 主训练循环（对照截图） =================
# 初始化优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

print("开始训练...")
for step in range(cfg.max_steps):
    # 1. 动态调整学习率
    lr = get_lr(step, cfg.max_steps)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    # 2. 采样数据
    x, y = dataset.sample_batch(cfg.batch_size)
    
    # 3. 前向传播
    logits = model(x) # (B, L, V)
    
    # 4. 计算 Loss（展平操作）
    # logits.view(-1, V) 将 (B, L, V) 变成 (B*L, V)
    # y.view(-1) 将 (B, L) 变成 (B*L)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    
    # 5. 反向传播 + 梯度裁剪 + 更新
    optimizer.zero_grad(set_to_none=True) # 比置零更省内存
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 限制梯度最大范数为1.0
    optimizer.step()
    
    # 6. 日志与存档
    if step % 50 == 0:
        ppl = math.exp(min(loss.item(), 20)) # 计算困惑度 Perplexity
        print(f"step {step:4d} | loss {loss.item():.4f} | ppl {ppl:.1f} | lr {lr:.2e}")
        
    if step % 250 == 0 and step > 0:
        torch.save(model.state_dict(), f"ckpt_step{step}.pt")
        print(f"--- 已保存模型存档 ckpt_step{step}.pt ---")
        