import torch
import torch.optim as optim
import torch.nn as nn



#####################################################################
# 逐步实现版本，用于数据类比和结构


# # 1. 数据：二分类玩具集
# xs = [[2.0, 3.0], [1.0, -1.0], [0.5, 0.5], [-1.0, 1.0]]
# ys = [1.0, -1.0, -1.0, 1.0]     # 标签 ±1

# # 2. 参数：w1, w2, b
# # 【对照】Value(0.1) -> torch.tensor(0.1, requires_grad=True)
# w1 = torch.tensor(0.1, requires_grad=True)
# w2 = torch.tensor(-0.2, requires_grad=True)
# b  = torch.tensor(0.0, requires_grad=True)

# for step in range(30):
#     total_loss = torch.tensor(0.0) # 【对照】Value(0.0)
    
#     for x, y in zip(xs, ys):
#         # 前向计算
#         # 【对照】w1 * Value(x[0]) -> w1 * torch.tensor(x[0])
#         score = w1 * torch.tensor(x[0]) + w2 * torch.tensor(x[1]) + b
#         pred = torch.tanh(score)                  # 【对照】score.tanh()
        
#         diff = pred - torch.tensor(y)             # 【对照】pred - Value(y)
#         total_loss = total_loss + diff * diff     # 【对照】拆分后相加
        
#     # 反向 + 手动 SGD 更新
#     # 【对照】for p in (w1, w2, b): p.grad = 0.0
#     w1.grad = None
#     w2.grad = None
#     b.grad = None
    
#     total_loss.backward() # 【对照】total_loss.backward()
    
#     lr = 0.05
#     # 【对照】w1.data -= lr * w1.grad
#     # 在 PyTorch 中，不能直接 w1 -= lr * w1.grad，因为这是原地操作且涉及计算图。
#     # 必须使用 no_grad() 或者直接修改 .data (不推荐，但和你的写法完全一致)
#     with torch.no_grad():
#         w1 -= lr * w1.grad
#         w2 -= lr * w2.grad
#         b  -= lr * b.grad
        
#     if step % 5 == 0:
#         print(f"step {step}: loss={total_loss.item():.4f}")



#####################################################################
# 工程实现版本，使用torch封装的对象化用法，这才是大规模实际使用的

print("use torch w1 w2 b")

# 数据转为 Tensor (统一 batch 维度)
xs = torch.tensor([[2.0, 3.0], [1.0, -1.0], [0.5, 0.5], [-1.0, 1.0]])
ys = torch.tensor([1.0, -1.0, -1.0, 1.0])

# 1. 参数：w1, w2, b
# 使用 nn.Parameter 包装，自动拥有 requires_grad=True
w1 = nn.Parameter(torch.tensor(0.1))
w2 = nn.Parameter(torch.tensor(-0.2))
b  = nn.Parameter(torch.tensor(0.0))

# 2. 优化器：把参数丢进去
optimizer = optim.SGD([w1, w2, b], lr=0.05)
loss_fn = nn.MSELoss() # 均方误差，对应 diff * diff 的 mean

for step in range(30):
    # 【清空梯度】
    optimizer.zero_grad()
    
    # 【前向计算】 (利用广播机制一次性算完 4 个样本)
    score = w1 * xs[:, 0] + w2 * xs[:, 1] + b
    pred = torch.tanh(score)
    
    # 【计算 Loss】
    loss = loss_fn(pred, ys)
    
    # 【反向传播】
    loss.backward()
    
    # 【更新参数】
    optimizer.step()
    
    if step % 5 == 0:
        print(f"step {step}: loss={loss.item():.4f}")




#####################################################################
# 更进一步工程实现版本，使用torch封装的nn网络 linear 等等


print("use nn.linear")

# 1. 准备数据 (明确指定 dtype 为 float32，防止报错)
xs = torch.tensor([[2.0, 3.0], 
                   [1.0, -1.0], 
                   [0.5, 0.5], 
                   [-1.0, 1.0]], dtype=torch.float32)  # 形状: (4, 2)
ys = torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float32) # 形状: (4,)

# 2. 定义模型层
# 输入特征维度是 2 (因为 xs 有 2 列)，输出维度是 1 (因为我们只预测一个 score)
layer = nn.Linear(in_features=2, out_features=1)

# 看看 PyTorch 底层帮你自动生成了什么参数
print("自动初始化的权重 W:", layer.weight)
print("自动初始化的偏置 b:", layer.bias)

# 3. 定义优化器和损失函数
# 直接把 layer 里的所有参数 (W 和 b) 丢给优化器
optimizer = optim.SGD(layer.parameters(), lr=0.05)
loss_fn = nn.MSELoss() 

# 4. 训练循环
for step in range(30):
    optimizer.zero_grad()
    
    # 【前向计算】
    # 这里发生了一次矩阵乘法：(4, 2) @ (2, 1) -> (4, 1)
    score = layer(xs) 
    
    # 因为 layer 输出是 (4, 1)，而标签 ys 是 (4,)，需要 squeeze 去掉多余的维度
    score = score.squeeze() 
    
    pred = torch.tanh(score)
    
    # 计算 Loss
    loss = loss_fn(pred, ys)
    
    # 【反向传播 & 更新参数】
    loss.backward()
    optimizer.step()
    
    if step % 5 == 0:
        print(f"step {step}: loss={loss.item():.4f}")


#####################################################################
# 更进一步工程实现版本，使用面向对象的前向传播，使用 nn.Module 封装，组合神经网络

print("use nn.Module")

# 数据同上
xs = torch.tensor([[2.0, 3.0], [1.0, -1.0], [0.5, 0.5], [-1.0, 1.0]], dtype=torch.float32)
ys = torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float32)

# 1. 定义模型
class SimpleNeuron(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义网络层
        # 第一层：线性层（有参数）
        self.linear = nn.Linear(2, 1)
        # 第二层：激活层（无参数）
        self.tanh = nn.Tanh() 
        
    def forward(self, x):
        # 显式定义数据的计算逻辑（这就是前向传播！）
        score = self.linear(x)
        pred = self.tanh(score)  # 逐网络计算调用
        return pred.squeeze()

# 2. 实例化模型、损失函数和优化器
model = SimpleNeuron()
optimizer = optim.SGD(model.parameters(), lr=0.05)
loss_fn = nn.MSELoss()

# 3. 训练循环
for step in range(30):
    optimizer.zero_grad()
    
    # 调用 model(xs)，实际上触发了 SimpleNeuron 中的 forward 函数
    pred = model(xs)
    
    loss = loss_fn(pred, ys)
    
    loss.backward()
    optimizer.step()
    
    if step % 5 == 0:
        print(f"step {step}: loss={loss.item():.4f}")


