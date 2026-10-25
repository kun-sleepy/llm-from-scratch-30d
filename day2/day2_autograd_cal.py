import torch

x = torch.tensor(2.0, requires_grad=True)   # 声明"我要对 x 求梯度"
y = x ** 2 + 3 * x                           # 前向：PyTorch 同时记录计算图
y.backward()                                 # 反向：从 y 出发链式法则回溯
print(x.grad)                                # dy/dx = 2x+3 = 7.0


