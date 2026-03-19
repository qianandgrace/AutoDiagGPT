import json
import random
import time
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from zhipuai import ZhipuAI

# ================== 配置 =================

INPUT_FILE = r"C:\Users\qian gao\git_project\AutoDiagGPT\data_prepare\instruction_data\alpaca_single.json"
OUTPUT_FILE = "car_finetune.json"

TARGET_SINGLE = 950
TARGET_MULTI = 50

SLEEP_TIME = 0.8
SIM_THRESHOLD = 0.9

client = ZhipuAI(api_key="982650cae3b54f9aa634e612c6a80eb7.4u6iqYAXPY66VUQv")
embed_model = SentenceTransformer("BAAI/bge-small-zh")
# ================== Embedding（可替换bge） ==================
def get_embedding(text):

    emb = embed_model.encode(text)
    return emb

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

embeddings = []

def is_duplicate(text):
    emb = get_embedding(text)

    for e in embeddings:
        if cosine_sim(emb, e) > SIM_THRESHOLD:
            return True

    embeddings.append(emb)
    return False


# ================== Prompt ==================
def build_prompt(item, mode="single"):
    base = f"""
你是一个有10年经验的汽车维修技师，请基于以下案例生成新的训练数据：

原始数据：
{json.dumps(item, ensure_ascii=False)}

要求：
1. 输入必须像真实车主描述（口语、不专业）
2. 可以加入：
   - 车型（如本田、丰田、大众）
   - 时间（最近、这几天）
   - 使用场景（堵车、高速、冷车启动）
3. 不要照抄原句，必须改写
4. 分析必须像维修记录（如：检查发现...）
5. 解决方案必须具体（更换/清洗/维修）
6. 输出内容控制在200字以内
"""

    if mode == "single":
        return base + """
请生成一条数据：

{
"instruction": "根据用户描述诊断汽车故障并给出维修建议",
"input": "...",
"output": "..."
}
只输出JSON
"""

    else:
        return base + """
请生成一条多轮对话数据：

{
"instruction": "...",
"input": "...",
"output": "...",
"history": [
  ["用户问题","助手回答"],
  ["用户问题","助手回答"]
]
}
只输出JSON
"""

def extract_json(text):
    if not text:
        return None

    # ===== 1. 去掉 ```json ``` =====
    text = text.strip()
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)

    # ===== 2. 找最外层 {} =====
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        return None

    text = text[start:end+1]

    # ===== 3. 尝试解析 =====
    try:
        return json.loads(text)
    except:
        return None


# ================== LLM调用 ==================
def call_llm(prompt):
    try:
        resp = client.chat.completions.create(
                model="glm-3-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                max_tokens=800
            )
        text = resp.choices[0].message.content.strip()

        data = extract_json(text)
        if not data:
            print("⚠️ 解析失败原文：", text[:200])
            return None

        return data

    except Exception as e:
        print("解析失败:", e)
        return None


# ================== 质量过滤 ==================
def is_valid(data, mode):
    if "input" not in data or "output" not in data:
        return False

    if len(data["input"]) < 10:
        return False

    if "建议" not in data["output"]:
        return False

    if mode == "multi":
        if "history" not in data or len(data["history"]) < 1:
            return False

    return True


# ================== 主流程 ==================
def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    single_count = 0
    multi_count = 0

    pbar = tqdm(total=TARGET_SINGLE + TARGET_MULTI)
    while single_count < TARGET_SINGLE or multi_count < TARGET_MULTI:
        item = random.choice(data)

        # ===== 控制比例 =====
        if single_count < TARGET_SINGLE:
            mode = "single"
        else:
            mode = "multi"

        prompt = build_prompt(item, mode)
        new_data = call_llm(prompt)

        if not new_data:
            continue

        # ===== 质量过滤 =====
        if not is_valid(new_data, mode):
            continue

        # ===== 去重 =====
        # if is_duplicate(new_data["input"]):
        #     continue

        # ===== 计数 =====
        if mode == "single":
            single_count += 1
        else:
            multi_count += 1

        results.append(new_data)
        pbar.update(1)

        time.sleep(SLEEP_TIME)

    # ===== 打乱 =====
    random.shuffle(results)

    # ===== 保存 =====
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n完成")
    print("单轮:", single_count)
    print("多轮:", multi_count)


if __name__ == "__main__":
    main()