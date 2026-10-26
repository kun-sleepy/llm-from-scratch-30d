
def gen():
    print("start")
    yield 1
    print("middle")
    yield 2
    print("end")
    yield 3

g = gen()          # 什么都不打印！只是创建生成器对象
print(type(g))     # <class 'generator'>

next(g)            # 打印 "start"，返回 1
next(g)            # 打印 "middle"，返回 2
# next(g)            # 打印 "end"，返回 3
# next(g)            # 抛 StopIteration




class Adder:
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        return x + self.n

add5 = Adder(5)
print(add5(10))     # 15   ← 实例被调用
print(callable(add5))  # True



class BatchSampler:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __call__(self, data):               # ← 含 yield，调用即返回生成器
        for i in range(0, len(data), self.batch_size):
            yield data[i : i + self.batch_size]

sampler = BatchSampler(3)
g = sampler(range(10))     # 返回 generator，未执行
print(type(g))             # <class 'generator'>

for batch in g:
    print(batch)
# [0, 1, 2]  [3, 4, 5]  [6, 7, 8]  [9]

g = sampler(range(12))     # 返回 generator，未执行
print(type(g))             # <class 'generator'>
for batch in g:
    print(batch)

