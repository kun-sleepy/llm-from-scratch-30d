
from graphviz import Digraph



class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data          # 前向值
        self.grad = 0.0           # 梯度（初始为 0）
        self._backward = lambda: None   # 该节点对父节点的局部反向传播
        self._prev = set(_children)     # 计算图中的父节点

        # 新增：用于可视化的标签
        self._op = _op 
        self.label = label  

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"


    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # 局部梯度 ∂out/∂self = 1，∂out/∂other = 1
            # 链式法则：下游梯度 = 上游梯度 × 局部梯度
            # 为什么 += 而不是 =？回忆昨天 b 的两条路径！一个节点可被多次使用
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    # --- 反向加法 (支持 3 + Value) ---
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
            other = other if isinstance(other, Value) else Value(other)
            out = Value(self.data - other.data, (self, other), '-')
    
            def _backward():
                # 局部梯度 ∂out/∂self = 1，∂out/∂other = 1
                # 链式法则：下游梯度 = 上游梯度 × 局部梯度
                # 为什么 += 而不是 =？回忆昨天 b 的两条路径！一个节点可被多次使用
                self.grad += 1.0 * out.grad # 被减数的梯度仍然为正
                other.grad += -1.0 * out.grad
            out._backward = _backward
            return out
        # --- 反向减法 (支持 3 + Value) ---
    def __rsub__(self, other):
        return Value(other) - self

    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # 预习考核第 1 题答案在此：∂out/∂self = other，∂out/∂other = self
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    # --- 反向乘法 (支持 3 * Value) ---
    def __rmul__(self, other):
        return self * other

    # def __pow__(self, other):
    #        # 正常计算x^2等是正确的，但是对幂的求导（x^y）没有做算法，所以最好还是直接约束为常数幂的函数
    #         other = other if isinstance(other, Value) else Value(other)
    #         out = Value(self.data ** other.data, (self, other), '*')
    
    #         def _backward():
    #             # 预习考核第 1 题答案在此：∂out/∂self = other，∂out/∂other = self
    #             self.grad += other.data*self.data ** (other.data-1)* out.grad
    #             other.grad += self.data * out.grad
    #         out._backward = _backward
    #         return out

    def __pow__(self, other):
        # 修正：仅支持常数次幂
        assert isinstance(other, (int, float)), "只支持 int/float 的幂"
        out = Value(self.data ** other, (self,), f'**{other}')
        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        import math
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')
        def _backward():
            # d(tanh)/dx = 1 - tanh² —— 纸笔推导一遍再写
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out


    def backward(self):
        # 拓扑排序：保证每个节点在其所有"孩子"处理完后再处理
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)

        self.grad = 1.0          # d(loss)/d(loss) = 1，反向传播的"种子"
        for node in reversed(topo):
            node._backward()     # 从输出往回，逐节点执行局部链式法则

def trace(root):
    # 遍历图，收集所有的节点和边
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    # 创建有向图，方向为从左到右 (LR)
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
    
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # 为每个 Value 节点创建一个记录形状的节点
        dot.node(name=uid, label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            # 如果有操作符（如 +, *），创建一个单独的节点代表操作
            dot.node(name=uid + n._op, label=n._op)
            # 连接操作符节点和输出节点
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # 连接输入节点到操作节点
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot


    
#####################################################################
# 用昨天的 T2 同款图对拍：
a, b = Value(3.0), Value(2.0)
c = a * b
print(type(c))
d = c + b          # 需要支持 Value + Value，以及 __radd__/__rmul__ 方便常数运算
print(type(d))
loss = d * d
loss.backward()
print(a.grad, b.grad)   # 应得 32 和 56？手算验证（b 仍是两条路径）



#####################################################################

# # 数据：二分类玩具集
# xs = [[2.0, 3.0], [1.0, -1.0], [0.5, 0.5], [-1.0, 1.0]]
# ys = [1.0, -1.0, -1.0, 1.0]     # 标签 ±1

# # 参数：w1, w2, b 都是 Value
# w1, w2, b = Value(0.1), Value(-0.2), Value(0.0)

# for step in range(30):
#     total_loss = Value(0.0)
#     for x, y in zip(xs, ys):
#         score = w1 * Value(x[0]) + w2 * Value(x[1]) + b
#         pred = score.tanh()                      # 输出压到 (-1,1)
#         total_loss = total_loss + (pred - Value(y)) * (pred - Value(y))  # 需要 __sub__/__pow__ 或拆解
#     # 反向 + 手动 SGD 更新
#     for p in (w1, w2, b): p.grad = 0.0
#     total_loss.backward()
#     lr = 0.05
#     w1.data -= lr * w1.grad
#     w2.data -= lr * w2.grad
#     b.data  -= lr * b.grad
#     if step % 5 == 0:
#         print(f"step {step}: loss={total_loss.data:.4f}")

#####################################################################

# 1. 定义输入和参数 (记得加上 label)
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
b = Value(6.8813735870195432, label='b')

# 2. 前向计算：模拟一个神经元
# x1*w1 + x2*w2 + b
x1w1 = x1 * w1; x1w1.label = 'x1*w1'
x2w2 = x2 * w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'

# 激活函数 tanh
o = n.tanh(); o.label = 'o'

# 3. 反向传播
o.backward()

# 4. 生成可视化图
dot = draw_dot(o)

# 在 Jupyter Notebook 中，直接写 dot 就能显示图片
# 如果你想保存为图片文件存进仓库：
dot.render('neuron_graph', view=True) 

