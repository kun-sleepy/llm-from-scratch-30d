import torch

x = torch.tensor(2.0, requires_grad=True)   # 声明"我要对 x 求梯度"
y = x ** 2 + 3 * x                           # 前向：PyTorch 同时记录计算图
y.backward()                                 # 反向：从 y 出发链式法则回溯
print(x.grad)                                # dy/dx = 2x+3 = 7.0


#####################################################################
# 图：a=3, b=2 → c=a*b → d=c+b → loss=d**2
# 你手算：∂loss/∂a = ?  ∂loss/∂b = ?
# （提示：∂loss/∂d=2d=2*8=16；∂d/∂c=1；∂c/∂a=b=2 → ∂loss/∂a=16*1*2=32）

a = torch.tensor(3.0, requires_grad=True)
b = torch.tensor(2.0, requires_grad=True)
c = a * b
d = c + b
loss = d ** 2
loss.backward()
print(a.grad, b.grad)   # 应输出 32.0 和 ...？（b 有两条路径！c 路径和 d 直连路径）


#####################################################################

# 任务：拟合 y = 3x + 2，只能用 backward() + 手动更新参数
torch.manual_seed(0)
x = torch.linspace(0, 1, 100)
y = 3 * x + 2 + 0.1 * torch.randn(100)

w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
lr = 0.5

for step in range(200):
    y_hat = w * x + b
    loss = ((y_hat - y) ** 2).mean()
    loss.backward()
    with torch.no_grad():          # 参数更新不参与计算图
        w -= lr * w.grad
        b -= lr * b.grad
    w.grad.zero_()                 # 手动清梯度！体会为什么需要它
    b.grad.zero_()
    if step % 50 == 0:
        print(f"step {step}: loss={loss.item():.4f}, w={w.item():.3f}, b={b.item():.3f}")


