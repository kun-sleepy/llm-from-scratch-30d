import numpy as np
import tiktoken
import os

def encode_file(file_path, output_npy_path, tokenizer_name="gpt2"):
    """
    读取文本文件，使用 tokenizer 进行编码，并保存为 numpy 数组 (.npy)
    """
    # 1. 初始化 Tokenizer
    enc = tiktoken.get_encoding(tokenizer_name)
    
    # 2. 读取文本内容 (假设是 utf-8 编码)
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 3. 将文本编码成 Token ID 列表
    tokens = enc.encode(text)
    
    # 4. 在文档末尾追加特殊分隔符 (GPT-2 的 eot_token 是 50256)
    tokens.append(enc.eot_token)
    
    # ⚠️【极其关键的坑：数据类型】
    # 截图里写 np.array(uint16)，但如果用 GPT-2 词表大小是 50257，
    # 超过了 uint16 的上限 (65535)吗？没超过，刚刚好。
    # 但如果是 Qwen 词表大小 151936，就必须用 np.uint32！
    # 这里我们用 uint16 演示，如果报错或溢出，请换成 uint32。
    tokens_np = np.array(tokens, dtype=np.uint16)
    
    # 5. 保存为 .npy 格式（二进制文件，读取速度极快）
    np.save(output_npy_path, tokens_np)
    print(f"✅ 编码完成！共 {len(tokens)} 个 token, 已保存至 {output_npy_path}")
    
    return enc

def test_round_trip(enc, test_text):
    """
    单元验收：测试 解码(编码(文本)) == 原始文本
    """
    print("\n--- 开始 Round-Trip 验收 ---")
    # 编码
    encoded_ids = enc.encode(test_text)
    # 解码
    decoded_text = enc.decode(encoded_ids)
    
    # 断言验证
    assert decoded_text == test_text, f"❌ 解码失败！\n原文: {test_text}\n解码: {decoded_text}"
    print("✅ 验收通过：解码后的文本与原始文本完全一致！")

# ================= 模拟执行 =================
if __name__ == "__main__":
    # 0. 造一个 1000 字的模拟文本
    long_text = "Deep learning is an important branch of artificial intelligence." * 60  # 约 1000 字
    test_file = "temp_text.txt"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(long_text)
        
    # 1. 执行 encode_file
    enc = encode_file(test_file, r"day6\data_tokens.npy", tokenizer_name="gpt2")
    
    # 2. 执行单元验收
    test_round_trip(enc, long_text)
    
    # 3. 验证 .npy 文件加载
    loaded_tokens = np.load(r"day6\data_tokens.npy")
    print(f"\n.npy 文件加载成功，形状: {loaded_tokens.shape}, 类型: {loaded_tokens.dtype}")
    
    # 清理临时文件
    os.remove(test_file)
