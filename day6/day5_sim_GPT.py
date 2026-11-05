from day5_transformer import *


class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_len):
        super().__init__()
        self.max_len = max_len 
        self.embed = nn.Embedding(vocab_size, d_model)   # 无位置 Embedding！RoPE 已接管
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, max_len) for _ in range(n_layers)
        ])
        self.norm_f = RMSNorm(d_model)                    # 最后的 Norm 不能省！
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight           # weight tying：输入输出共享权重

    def forward(self, idx):
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.norm_f(x))               # (B, L, vocab)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.max_len:])[:, -1, :] / temperature   # 只看最后一个位置
            next_id = torch.multinomial(torch.softmax(logits, -1), 1)        # 采样而非 argmax
            idx = torch.cat([idx, next_id], dim=1)
        return idx

model = GPT(vocab_size=1000, d_model=512, n_heads=8, d_ff=2048, n_layers=4, max_len=128)
# 注意 register_buffer 需要：__init__ 里 self.max_len = max_len
idx = torch.randint(0, 1000, (2, 16))
logits = model(idx)
assert logits.shape == (2, 16, 1000)
# 未训练的 generate 应能跑通（输出乱码无所谓，流程必须通）
out = model.generate(idx[:, :4], max_new_tokens=10)
assert out.shape == (2, 14)
# 整机参数量手算对拍（写在日志里，面试直接背）：
# Embedding 1000×512（共享一次） + 4×4.2M + 最终 norm 512 ≈ 17.3M

