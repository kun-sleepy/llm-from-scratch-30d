import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time  # 引入时间模块用于记录耗时

print(torch.__version__)
print(torch.cuda.is_available()) 
#  device：张量住在哪个设备上
device = 'cuda' if torch.cuda.is_available() else 'cpu'



train_data = datasets.MNIST(root='data', train=True, download=True,
                            transform=transforms.ToTensor())
test_data  = datasets.MNIST(root='data', train=False, download=True,
                            transform=transforms.ToTensor())

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)   # ← 昨天的生成器！
test_loader  = DataLoader(test_data, batch_size=256)

x, y = next(iter(train_loader))
print(x.shape, y.shape)   # (64, 1, 28, 28) 和 (64,)——batch 维在最前



class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),              # (64,1,28,28) → (64, 784)
            nn.Linear(784, 256),
            nn.ReLU(),                 # ← 昨天 x[x<0]=0 的张量版
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10),         # 10 个数字类别
        )
    def forward(self, x):
        return self.net(x)             # 输出 logits（未归一化分数）


results = []  # 用于保存结果

for name, opt_fn in [('SGD',  lambda p: torch.optim.SGD(p, lr=0.01)),
                     ('SGD+mom', lambda p: torch.optim.SGD(p, lr=0.01, momentum=0.9)),
                     ('AdamW', lambda p: torch.optim.AdamW(p, lr=1e-3))]:

    model = MLP().to(device)
    print(sum(p.numel() for p in model.parameters()))   # 手算验证参数量！


    loss_fn = nn.CrossEntropyLoss()                        # 内置 softmax，直接吃 logits

    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    print(f"method is {name}")
    optimizer = opt_fn(model.parameters())

     # 记录开始时间
    start_time = time.time()

    for epoch in range(2):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)                # ① 前向
            loss = loss_fn(pred, y)        # ② 算损失
            optimizer.zero_grad()          # ③ 清梯度（回忆上午的累加实验）
            loss.backward()                # ④ 反向传播
            optimizer.step()               # ⑤ 更新参数
        # 每个 epoch 后评估一次
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                pred = model(x.to(device)).argmax(dim=1)
                correct += (pred == y.to(device)).sum().item()
                total += len(y)
        print(f"epoch {epoch}: test acc = {correct/total:.4f}")

    # 记录结束时间并计算耗时
    train_time = time.time() - start_time
    # --- 评估阶段 ---
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x.to(device)).argmax(dim=1)
            correct += (pred == y.to(device)).sum().item()
            total += len(y)
            
    acc = correct / total
    results.append((name, acc, train_time))
    print(f"✅ {name} 完成 | 准确率: {acc:.4f} | 耗时: {train_time:.2f} 秒")

