import gymnasium as gym
import numpy as np

# 1. 创建游戏 (环境)
env = gym.make("CartPole-v1")    # 倒立摆！你的老熟人
obs, info = env.reset()          # obs = [小车位置, 速度, 摆角, 角速度]
# 2. 查看状态和动作的空间 (Space)
print(env.observation_space, env.action_space)   # 连续状态 Box(4)，离散动作 {左推, 右推}

# 4. 手工循环（随机瞎蒙）
total_r, trajectory = 0, []  # total_r 累计得分，trajectory 存下整盘游戏的记忆
for t in range(500):
    # 4.1 动作采样
    a = env.action_space.sample() 
    # 因为这是一个初始的“随机策略”，它就在 [0, 1] 里随机抽一个。
    
    # 4.2 与环境交互
    obs, r, terminated, truncated, _ = env.step(a) 
    # 【核心步骤】把动作 a 告诉环境，环境算出下一帧 obs，给奖励 r，并告诉你游戏结束没。
    
    # 4.3 记录
    trajectory.append((obs, a, r))
    total_r += r
    
    # 4.4 判断是否结束
    if terminated or truncated:
        break

print(f"随机策略存活 {t+1} 步，回报 {total_r}")     # 通常 20~40 步就倒
