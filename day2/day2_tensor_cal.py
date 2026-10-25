import numpy as np
import torch
print(torch.__version__)
print(torch.cuda.is_available())   # False 也没关系，前两周 CPU 足够

# ② device：张量住在哪个设备上
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# # 创建：API 几乎一一对应
# np.zeros((3,4))          →  torch.zeros(3, 4)
# np.random.randn(3,4)     →  torch.randn(3, 4)
# np.arange(10)            →  torch.arange(10)
# np.array([[1,2],[3,4]])  →  torch.tensor([[1,2],[3,4]])

# # 运算：名字基本一样
# A @ B        →  A @ B          # 矩阵乘写法相同
# A * B        →  A * B          # 逐元素乘，广播规则完全相同！
# A.mean(0)    →  A.mean(dim=0)  # 注意：axis → dim
# A.T          →  A.T 或 A.mT
# A.reshape    →  A.reshape 或 A.view

# # 三个高频差异（各写一行验证）：
# a = torch.randn(3, 4)
# a.sum()            # 返回 0 维张量，要 .item() 才取出 Python 数
# a.T.shape          # 转置行为一致
# torch.einsum('ij,jk->ik', a, a.T)   # einsum 语法与 NumPy 完全一致——昨天的练习直接保值

#####################################################################

# # ① dtype：PyTorch 默认 float32（NumPy 是 float64），神经网络都用 float32
# x = torch.randn(3)        # torch.float32
# x = torch.randn(3, dtype=torch.float64)  # 显式指定

# # ② device：张量住在哪个设备上
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# x = x.to(device)          # 搬到 GPU；两个张量运算前必须在同一 device！

# # ③ 与 NumPy 互转（注意共享内存陷阱——呼应昨天的"视图"题）
# t = torch.from_numpy(np_arr)   # 共享内存！改一个另一个跟着变
# n = t.numpy()                  # 反向同理；需要独立副本用 .clone()

#####################################################################


tt_tensor=torch.rand(1000,10)
print(1)
tt_tensor=tt_tensor.to(device=device)

tt_tensor_sm=tt_tensor.softmax(dim=1)

print(tt_tensor[0,:].sum())
print(tt_tensor_sm[0,:].sum())

