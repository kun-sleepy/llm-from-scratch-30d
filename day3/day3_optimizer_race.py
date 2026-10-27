import numpy as np
import matplotlib.pyplot as plt

# 目标函数：一个极度狭长的椭圆峡谷
def f(p):        
    return p[0]**2 + 50*p[1]**2

def grad_f(p):   
    return np.array([2*p[0], 100*p[1]])

def optimize(grad_f, p0, lr, steps, beta1=0.9, beta2=0.999, mode='sgd'):
    p = p0.copy().astype(float)
    m = np.zeros(2)
    v = np.zeros(2)
    traj = [p.copy()]
    
    for t in range(1, steps+1):
        g = grad_f(p)
        
        if mode == 'sgd':
            p -= lr * g
            
        elif mode == 'momentum':
            m = beta1 * m + (1 - beta1) * g
            p -= lr * m
            
        elif mode == 'rmsprop':
            # RMSprop 只维护二阶矩（无动量）
            v = beta2 * v + (1 - beta2) * g**2
            p -= lr * g / (np.sqrt(v) + 1e-8)
            
        elif mode == 'adam':
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g**2
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
            
        traj.append(p.copy())
    return np.array(traj)

# 初始点
p0 = np.array([1.5, 0.3])
steps = 150

# 跑四条轨迹
# 注意：针对各自特性微调了学习率，让曲线更好看
traj_sgd  = optimize(grad_f, p0, lr=0.015, steps=steps, mode='sgd')
traj_mom  = optimize(grad_f, p0, lr=0.015, steps=steps, mode='momentum')
traj_rms  = optimize(grad_f, p0, lr=0.05,  steps=steps, mode='rmsprop')
traj_adam = optimize(grad_f, p0, lr=0.1,   steps=steps, mode='adam')

# ================= 可视化 =================
# 1. 生成网格用于画等高线
x = np.linspace(-1.8, 1.8, 200)
y = np.linspace(-0.6, 0.6, 200)
X, Y = np.meshgrid(x, y)
Z = X**2 + 50*Y**2

plt.figure(figsize=(12, 6))

# 画等高线（对数比例能更好展示峡谷形状）
levels = np.logspace(-1, 2, 15)
plt.contour(X, Y, Z, levels=levels, cmap='Greys', alpha=0.5, zorder=1)

# 2. 画四条轨迹
plt.plot(traj_sgd[:, 0], traj_sgd[:, 1], 'o-', label='SGD (lr=0.015)', color='blue', markersize=2, alpha=0.8, zorder=2)
plt.plot(traj_mom[:, 0], traj_mom[:, 1], 'o-', label='Momentum (lr=0.015)', color='orange', markersize=2, alpha=0.8, zorder=3)
plt.plot(traj_rms[:, 0], traj_rms[:, 1], 'o-', label='RMSprop (lr=0.05)', color='green', markersize=2, alpha=0.8, zorder=4)
plt.plot(traj_adam[:, 0], traj_adam[:, 1], 'o-', label='Adam (lr=0.1)', color='red', markersize=2, alpha=0.8, zorder=5)

# 标注起点和终点
plt.plot(p0[0], p0[1], 'k*', markersize=15, label='Start')
plt.plot(0, 0, 'kX', markersize=12, label='Global Minimum')

# 3. 设置图表格式
plt.title('Optimizer Race on Ill-conditioned Terrain', fontsize=14)
plt.xlabel('Parameter \(p_0\) (Flat direction)', fontsize=12)
plt.ylabel('Parameter \(p_1\) (Steep direction)', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()

# 保存图片
plt.savefig('optimizer_race.png', dpi=150)
plt.show()
