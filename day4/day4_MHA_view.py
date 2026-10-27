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






# 任务：输入序列 [a, b, c, ...]，让模型学会输出同样序列（identity）
# 用 1 层 MHA + 因果 mask 训练 200 步即可收敛
vocab, L, D, H = 20, 12, 64, 4
model = nn.Sequential(nn.Embedding(vocab, D))
mha = MultiHeadAttention(D, H)
head = nn.Linear(D, vocab)
opt = torch.optim.AdamW(list(model.parameters())+list(mha.parameters())+list(head.parameters()), lr=3e-3)

mask = torch.tril(torch.ones(L, L, dtype=torch.bool))
for step in range(300):
    seq = torch.randint(0, vocab, (32, L))
    h = model(seq)
    h, attn = mha(h, mask=mask)
    logits = head(h)
    loss = F.cross_entropy(logits.reshape(-1, vocab), seq.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 60 == 0: print(step, loss.item())


# 生成测试数据
seq_test = torch.randint(0, vocab, (32, L))
h_test = model(seq_test)
h_test, attn_test = mha(h_test, mask=mask)

# 画图
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for h in range(4):
    # 取出第0个样本、第h个头的注意力矩阵
    attn_matrix = attn_test[0, h].detach().cpu().numpy()
    
    ax = axes[h]
    im = ax.imshow(attn_matrix, cmap='viridis') # 选择热力图配色，viridis 或 hot 都行
    ax.set_title(f"Head {h}")
    ax.set_xlabel("Key (Source)")
    ax.set_ylabel("Query (Target)")
    
    # 加上数值，如果矩阵太大就不加了，这里 12x12 刚好可以加
    # for i in range(attn_matrix.shape[0]):
    #     for j in range(attn_matrix.shape[1]):
    #         ax.text(j, i, f"{attn_matrix[i, j]:.2f}", ha="center", va="center", color="w" if attn_matrix[i,j] < 0.5 else "k")

plt.tight_layout()
plt.show()



