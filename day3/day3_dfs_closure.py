
# 1. 这是一个普通的类
class Node:
    def __init__(self, val):
        self.val = val

# 2. 这是一个普通的函数
def say_hello():
    print("我是一个函数对象！")

# 3. 核心演示：像塞数字一样，把函数塞进对象
n = Node(10)
n.my_func = say_hello   # 没有报错！因为函数也是对象

# 调用看看
n.my_func()             # 输出: 我是一个函数对象！


def make_multiplier(x):
    # 内部函数
    def multiply(y):
        return x * y
    return multiply

# 第一次调用：x=2，返回了内部的 multiply 函数
double = make_multiplier(2)

# 第二次调用：x=3，返回了另一个内部的 multiply 函数
triple = make_multiplier(3)

# 此时外层函数早就执行完了，x 理论上应该被销毁了
# 但是！闭包把 x=2 和 x=3 分别“冻结”在了 double 和 triple 里
print(double(5))  # 输出: 10 (2*5)
print(triple(5))  # 输出: 15 (3*5)



class Node:
    def __init__(self, val):
        self.val = val
        self.grad = 0.0      # 梯度（导数）默认为 0

    # 魔法方法：当两个 Node 相乘时触发 a * b
    def __mul__(self, other):
        # 1. 前向计算：立刻算出当前的结果
        out = Node(self.val * other.val)
        
        # 2. 定义内部函数（闭包）：用来计算反向传播（梯度）
        def _backward():
            # 闭包魔法：这里用到了 self, other, out，它们会被自动“冻结”保存
            self.grad  += other.val * out.grad
            other.grad += self.val * out.grad
        
        # 3. 一切皆对象：把 _backward 这个函数，挂载到结果 out 对象上！
        out._backward = _backward
        
        # 4. 返回结果对象
        return out

# ===== 实际应用场景：构建计算图 =====
a = Node(2.0)
b = Node(3.0)

# 此时发生了两件事：
# 1. 算出了 c.val = 6.0 (前向计算)
# 2. c 对象身上多了一个名叫 _backward 的函数属性，且记住了 a, b, c 的值 (闭包)
c = a * b 

# 假设现在我们需要根据 c 的结果，求 a 和 b 的梯度（类似神经网络）
c.grad = 1.0          # 假设最终损失是 c，它对自己的导数是 1

# 关键：这就叫“延迟计算”！刚才 c = a * b 时并没有算梯度，
# 现在才触发刚才被塞进对象里的 _backward 函数
c._backward()         # 调用它身上挂载的函数

print(a.grad)         # 输出: 3.0 (因为 b=3)
print(b.grad)         # 输出: 2.0 (因为 a=2)


#####################################################################
# 1. 准备全局状态（通常放在函数外面，或者作为参数传入）
# visited = set()  # 记录走过的节点
# result = []      # 记录最终结果

# def dfs(node):
#     # 2. 边界条件 / 终止条件（撞南墙）
#     if node not in visited:
#         return
#     # 或者 if node is None: return
    
#     # 3. 标记当前节点（我去过这了）
#     visited.add(node)
    
#     # 4. 遍历所有的分岔路（递的过程）
#     for next_node in node.children:
#         dfs(next_node)  # 递归进入下一层
        
#     # 5. 处理当前节点（归的过程，后序操作）
#     # 这行代码决定了你是前序（自顶向下）还是后序（自底向上）
#     result.append(node) 


def count_down(n):
    # 1. 基线条件（Base Case）：阻止无限递归的刹车
    if n == 0:
        print("火箭发射！")
        return
    
    # 2. 执行当前层的任务
    print(f"倒计时: {n}")
    
    # 3. 递推（调用自己），这就是“深度优先”往下走
    count_down(n - 1)
    
    # 4. 这行代码在递归返回后才会执行（回溯时的动作）
    print(f"回到第 {n} 层")

count_down(3)


class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def dfs_search(root, target):
    # 1. 基线条件（走到空节点，或者找到了目标）
    if root is None:
        return False
    if root.val == target:
        return True
    
    # 2. 递推：尝试所有分支（左和右）
    # 只要左边或者右边有一条路走通了，就返回 True
    left_result = dfs_search(root.left, target)
    right_result = dfs_search(root.right, target)
    
    # 3. 汇总结果（因为左右都没找到，只能返回 False）
    return left_result or right_result


def find_all_paths(graph, start, end):
    result = []      # 外部变量，用来存所有找到的路径
    path = []        # 外部变量，记录当前正在走的路径

    # 内部函数：真正的 DFS 递归逻辑
    def dfs(node):
        # 1. 把当前节点加入路径（做选择）
        path.append(node)
        
        # 2. 基线条件：到达终点
        if node == end:
            result.append(path.copy()) # 注意：必须 copy！否则后面改 path 会影响到这里
        else:
            # 3. 递推：去邻居那里看看
            for neighbor in graph.get(node, []):
                if neighbor not in path: # 防止走回头路
                    dfs(neighbor)
                    
        # 4. 恢复现场（撤销选择）—— 这是 DFS 回溯的灵魂！
        path.pop() 

    # 启动递归
    dfs(start)
    return result

# 构建一个简单的图（字典表示邻接表）
graph = {
    'A': ['B', 'C'],
    'B': ['D'],
    'C': ['D'],
    'D': ['E'],
}
print(find_all_paths(graph, 'A', 'E'))
# 输出: [['A', 'B', 'D', 'E'], ['A', 'C', 'D', 'E']]



# DFS的两种场景和实现，注意区分和学习理解
# 新手在学习 DFS 时，最容易犯的致命错误就是“不分场景，把 path 当 visited 用”，或者“在需要回溯的地方忘了 pop”。
# 为了让你一眼看透，我们把这两种场景比作“扫地机器人”和“探险家走迷宫”。

# 扫地机器人去打扫房间。只要进过一个房间，它就在地图上打个勾（加入 visited）。以后永远不会再进这个房间。它不需要退回来的时候把记号擦掉。
# 探险家走迷宫，要在岔路口做标记（加入 path）。如果他走进死胡同，他必须原路退回来，
# 并且把死胡同的标记擦掉。只有这样，他换一条路走的时候，拿到的才是“纯净的当前路线”。 填数独





