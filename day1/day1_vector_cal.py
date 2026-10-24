import numpy as np
import time

N = 1_000_000
a = list(range(N))
b = list(range(N))

# 纯 Python 循环
t0 = time.time()
c = [x + y for x, y in zip(a, b)]
t1 = time.time()

# NumPy 向量化
na, nb = np.arange(N), np.arange(N)
t2 = time.time()
nc = na + nb
t3 = time.time()

print(f"Python 循环: {t1-t0:.3f}s   NumPy: {t3-t2:.4f}s   加速: {(t1-t0)/(t3-t2):.0f} 倍")
