import json
import os
import time
import random
import csv

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# ===== 初始化 =====
client = OpenAI(
    api_key=os.getenv("LAOZHANG_API_KEY"),  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
    base_url="https://api.laozhang.ai/v1"  # 
)
embed_model = SentenceTransformer("BAAI/bge-small-zh")
TOTAL_OBD = 300
TOTAL_CSV = 300
SIM_THRESHOLD = 0.95

embeddings = []
results = []

# ===== 相似度函数 =====
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def is_duplicate(text):
    emb = embed_model.encode(text)

    for e in embeddings:
        sim = cosine_similarity(emb, e)
        if sim > SIM_THRESHOLD:
            return True

    embeddings.append(emb)
    return False


# ===== LLM调用 =====
def call_llm(messages):
    response = client.chat.completions.create(
        model="gpt-4o-2024-05-13",
        messages=messages,
        temperature=0.8
    )

    text = response.choices[0].message.content.strip()

    try:
        return json.loads(text)
    except:
        print("解析失败:", text)
        return None


# ===== OBD生成 =====
def build_obd_messages(obd):
    example_user = "故障码: P0301\n症状: ['发动机抖动','加速无力']"
    example_assistant = """
{
  "input": "我这车一踩油门就抖得厉害，还报码P0301，是不是发动机出问题了？",
  "output": {
    "fault_code": "P0301",
    "analysis": "检查发现1缸点火不稳定，火花塞有积碳",
    "solution": "更换火花塞并清理积碳后恢复正常"
  }
}
"""

    return [
        {
            "role": "system",
            "content": """
你是汽车维修技师，请生成真实维修案例。

要求：
1. input必须是用户口语
2. 可以包含故障码，也可以不包含
3. 加入车型（随机）
4. output像维修记录
5. fault_code可以没有（不要强行生成）
6. 只输出JSON
"""
        },
        {"role": "user", "content": example_user},
        {"role": "assistant", "content": example_assistant},
        {
            "role": "user",
            "content": f"""
故障码: {obd.get("code")}
症状: {random.sample(obd.get("symptoms", []), min(2, len(obd.get("symptoms", []))))}
原因: {obd.get("possible_causes")}
"""
        }
    ]


# ===== CSV生成 =====
def build_csv_messages(row):
    example_user = "故障描述: 发动机抖动\n原因分析: 火花塞老化\n解决措施: 更换火花塞"
    example_assistant = """
{
  "input": "我这辆大众最近冷车启动抖得厉害，开起来也没劲，是不是哪里有问题？",
  "output": {
    "fault_code": "不涉及故障灯",
    "analysis": "检查后发现火花塞老化严重，点火不稳定",
    "solution": "更换火花塞后问题解决"
  }
}
"""

    return [
        {
            "role": "system",
            "content": """
你是资深汽车维修技师。

要求：
1. 故障描述要像车主说的话
2. 加入车型（随机）
3. 场景真实（冷车/热车/高速）
4. 不一定有故障码
5. 故障码不一定有具体的码，可以形容为“发动机故障灯亮”等等
6. 只输出JSON
"""
        },
        {"role": "user", "content": example_user},
        {"role": "assistant", "content": example_assistant},
        {
            "role": "user",
            "content": f"""
故障描述: {row['故障描述']}
原因分析: {row['原因分析']}
解决措施: {row['解决措施']}
"""
        }
    ]


# ===== 主流程 =====
def main():
    global results
    valid_data_num = 0
    # ===== 读取OBD =====
    with open(r"C:\Users\qian gao\git_project\AutoDiagGPT\data_prepare\rag_data\obd_codes.json", "r", encoding="utf-8") as f:
        obd_data = json.load(f)
    
    # ===== 读取CSV =====
    repair_data = []
    with open(r"C:\Users\qian gao\git_project\AutoDiagGPT\data_prepare\origin_data\repair_origin_data.csv", "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            repair_data.append(row)
    # ===== 1. OBD生成 =====
    for i in range(TOTAL_OBD):
        obd = random.choice(obd_data)

        print("OBD:", obd["code"])

        data = call_llm(build_obd_messages(obd))
        if data and not is_duplicate(data["input"]):
            valid_data_num += 1
            results.append(data)
        print(f"当前有效数据量: {valid_data_num}")
        time.sleep(0.8)

    # ===== 2. CSV生成 =====
    for i in range(TOTAL_CSV):
        row = random.choice(repair_data)

        print("CSV:", row["故障描述"])

        data = call_llm(build_csv_messages(row))
        if data and not is_duplicate(data["input"]):
            valid_data_num += 1
            results.append(data)
        print(f"当前有效数据量: {valid_data_num}")
        time.sleep(0.8)

    # ===== 保存 =====
    with open("final_dataset.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("完成，总数据量:", len(results))


if __name__ == "__main__":
    main()