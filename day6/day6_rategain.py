import math

def get_lr(step, warmup=300, total=5000, lr_max=3e-4, lr_min=3e-5):
    if step < warmup:                                  # 线性爬升（Adam 二阶矩未稳，防开局爆冲）
        return lr_max * (step + 1) / warmup
    if step >= total:
        return lr_min
    ratio = (step - warmup) / (total - warmup)
    return lr_min + 0.5 * (1 + math.cos(math.pi * ratio)) * (lr_max - lr_min)  # cosine 衰减

# 可视化确认形状（先画出来再训练——养成"先验证曲线形状"的习惯）
import matplotlib.pyplot as plt
lrs = [get_lr(i) for i in range(5000)]
plt.plot(lrs); plt.title("warmup + cosine decay"); plt.savefig(r"day6\lr_schedule.png")
