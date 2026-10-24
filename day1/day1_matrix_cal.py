import numpy as np

A = np.arange(6).reshape(2, 3)
B = np.arange(6).reshape(3, 2)

A * B.T       # 逐元素乘（Hadamard 积）——形状必须能广播
A @ B         # 矩阵乘（推荐写法，等价 np.matmul）→ (2,2)
np.dot(A, B)  # 二维时同 @，但高维行为不同，容易混乱，建议忘掉它

#####################################################################

# 基础：矩阵乘法
C = np.einsum('ij,jk->ik', A, B)      # 等价 A @ B
# 读法：A 的第 j 维和 B 的第 j 维"同名字母→对应相乘并求和"，i,k 保留

# 进阶：batch 矩阵乘（1000 条轨迹同时乘 A）
x = np.random.randn(1000, 2)
Ax = np.einsum('ij,bj->bi', A_mat, x)  # b 是 batch 维，不参与求和

# 重点：Attention 打分公式 score = Q·Kᵀ
Q = np.random.randn(2, 4, 8)   # (batch=2, heads=4, dim=8) —— 简化版
K = np.random.randn(2, 4, 8)
scores = np.einsum('bhd,bhe->bhe', Q, K)   # d 与 e 是"查询-键"的配对维
# 面试手写 Attention 时这一行就是核心

#####################################################################
a = np.random.randn(3, 4); b = np.random.randn(4, 5); v = np.random.randn(4)

# 1. a @ b              → einsum('??,??->??')
# 2. a @ v（矩阵乘向量） → einsum('??,?->??')
# 3. (a * a).sum() 逐元素平方和 → einsum('??,??->')


#####################################################################


np.einsum('ij,jk->ik', a, b)
np.einsum('ij,j->i', a, v)
np.einsum('ij,ij->', a, a)

