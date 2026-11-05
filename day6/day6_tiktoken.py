import tiktoken
enc = tiktoken.get_encoding("gpt2")      # 50,257 词表的 BPE

text = "The controller stabilizes the inverted pendulum."
ids = enc.encode(text)
print(ids)                                # 如 [464, 15331, 44962, ...]
print([enc.decode([i]) for i in ids])     # 看切分：常见词整块，生僻词拆成子词
# 观察实验：
ids=enc.encode("antidisestablishmentarianism")  # 超长词被拆成多个子词——BPE 的"折中"
print(ids)                                # 如 [464, 15331, 44962, ...]
print([enc.decode([i]) for i in ids])     # 看切分：常见词整块，生僻词拆成子词
ids=enc.encode("状态空间方程")                    # 中文会怎样？（GPT-2 词表没训过中文，会拆得很碎）
print(ids)                                # 如 [464, 15331, 44962, ...]
print([enc.decode([i]) for i in ids])     # 看切分：常见词整块，生僻词拆成子词




