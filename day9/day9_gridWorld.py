import numpy as np

N = 4          # 4x4 网格
gamma = 0.99   # 折扣因子

# 定义环境的运作规则：给定当前格子 s 和动作 a，返回 (下一个格子, 奖励, 是否结束)
def step(s, a):
    r, c = divmod(s, N) # s 是 0-15，divmod 可以把一维索引变成 (行, 列)
    if a == 0: r = max(r-1, 0)      # 0: 向上走 (如果已在第0行，就原地不动)
    if a == 1: r = min(r+1, N-1)    # 1: 向下走
    if a == 2: c = max(c-1, 0)      # 2: 向左走
    if a == 3: c = min(c+1, N-1)    # 3: 向右走
    
    s2 = r * N + c                  # 把 (行, 列) 重新压回一维索引
    reward = -1 if s2 != 15 else 0  # 只要没到终点（15），每步扣 1 分；到了终点 0 分
    done = (s2 == 15)               # 是否到达终点
    return s2, reward, done

# 初始化每个格子的身价 V 为 0
V = np.zeros(N * N)

# --- 核心：值迭代 (Value Iteration) ---
for it in range(100): # 最多迭代 100 轮
    delta = 0         # 记录这一轮里，格子身价变化的最大幅度
    for s in range(N * N): # 遍历 0 到 15 的每一个格子
        if s == 15: continue # 终点不需要更新
        
        v_old = V[s]       # 记住更新前的身价
        
        # 核心公式实现：找四个动作中，能带来最大身价的那一个
        # step(s, a)[1] 是奖励 r
        # step(s, a)[0] 是下一个格子 s2
        V[s] = max( step(s, a)[1] + gamma * V[step(s, a)[0]] for a in range(4) )
        
        # 记录本次更新产生的最大变化量
        delta = max(delta, abs(v_old - V[s]))
        
    # 如果所有格子的身价变化都极小，说明已经收敛了
    if delta < 1e-6:
        print(f"✅ 收敛于第 {it} 轮")
        break

# 打印 4x4 的身价场
print(V.reshape(N, N).round(1))
