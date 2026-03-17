import csv
import os
import json
import time
from openai import OpenAI

# 初始化客户端（替换成你的key）
client = OpenAI(
    api_key=os.getenv("LAOZHANG_API_KEY"),  # 如果您没有配置环境变量，请在此处用您的API Key进行替换
    base_url="https://api.laozhang.ai/v1"  # 百炼服务的base_url
)

INPUT_CSV = r"C:\Users\qian gao\git_project\AutoDiagGPT\data_prepare\top_300_obd_codes.csv"
OUTPUT_JSON = "obd_codes.json"


# ===== 全局统计 =====
total_prompt_tokens = 0
total_completion_tokens = 0
total_tokens = 0

def generate_obd_record(code, desc):
    global total_prompt_tokens, total_completion_tokens, total_tokens
    messages = [
        {
            "role": "system",
            "content": """你是资深汽车维修工程师，擅长OBD故障诊断，请输出结构化、专业、简洁的中文维修知识。要求：
                        1. 只输出JSON
                        2. 不要解释
                        3. 每个字段必须合理
                        4. 不要编造不存在的专业名词"""
        },

        # ===== 示例1 =====
        {
            "role": "user",
            "content": "故障码: P0301\n描述: Cylinder 1 Misfire Detected"
        },
        {
            "role": "assistant",
            "content": """
{
  "code": "P0301",
  "description": "发动机1缸失火",
  "symptoms": ["发动机抖动", "加速无力", "油耗增加"],
  "possible_causes": ["火花塞故障", "点火线圈损坏", "喷油嘴堵塞"],
  "solutions": ["更换火花塞", "检查或更换点火线圈", "清洗或更换喷油嘴"]
}
"""
        },

        # ===== 示例2 =====
        {
            "role": "user",
            "content": "故障码: P0171\n描述: System Too Lean (Bank 1)"
        },
        {
            "role": "assistant",
            "content": """
{
  "code": "P0171",
  "description": "系统过稀（第1排）",
  "symptoms": ["发动机动力不足", "怠速不稳", "油耗增加"],
  "possible_causes": ["进气系统漏气", "空气流量计故障", "燃油压力不足"],
  "solutions": ["检查进气系统是否漏气", "清洗或更换空气流量计", "检查燃油泵和燃油压力"]
}
"""
        },

        # ===== 示例3 =====
        {
            "role": "user",
            "content": "故障码: P0420\n描述: Catalyst System Efficiency Below Threshold (Bank 1)"
        },
        {
            "role": "assistant",
            "content": """
{
  "code": "P0420",
  "description": "催化转化器效率低于阈值（第1排）",
  "symptoms": ["发动机故障灯亮", "排放超标", "动力下降"],
  "possible_causes": ["三元催化器老化", "氧传感器故障", "燃烧不充分"],
  "solutions": ["更换三元催化器", "检查或更换氧传感器", "检查发动机燃烧状态"]
}
"""
        },

        # ===== 真实输入 =====
        {
            "role": "user",
            "content": f"故障码: {code}\n描述: {desc}"
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o-2024-05-13",
        messages=messages,
        temperature=0.2,  # 更稳定
    )
    # ===== Token统计 =====
    usage = response.usage

    total_prompt_tokens += usage.prompt_tokens
    total_completion_tokens += usage.completion_tokens
    total_tokens += usage.total_tokens

    print(f"[{code}] tokens -> prompt: {usage.prompt_tokens}, completion: {usage.completion_tokens}, total: {usage.total_tokens}")
    text = response.choices[0].message.content.strip()

    try:
        return json.loads(text)
    except:
        print("解析失败:", text)
        return None

def main():
    results = []

    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):

            code = row["code"]
            desc = row["description"]

            print(f"处理 {code}...")

            data = generate_obd_record(code, desc)
            if data:
                results.append(data)

            time.sleep(1)  # 防止限流

            # 👉 测试时只跑前10条
            # if i >= 10:
            #     break

    # 保存 JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("完成，已保存到", OUTPUT_JSON)


if __name__ == "__main__":
    main()