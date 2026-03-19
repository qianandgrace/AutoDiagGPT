import json
import random

INPUT_FILE = r"C:\Users\qian gao\git_project\AutoDiagGPT\data_prepare\rag_data\final_dataset.json"
OUTPUT_SINGLE = "alpaca_single.json"
OUTPUT_MULTI = "alpaca_multi.json"

# ===== 指令模板 =====
INSTRUCTIONS = {
    "diagnosis": "根据用户描述诊断汽车故障并给出维修建议",
    "explain": "解释汽车故障或故障码的含义",
    "advice": "判断车辆是否可以继续行驶并给出建议",
    "chat": "进行汽车故障诊断对话"
}

# ===== 单轮生成 =====
def build_single(item):
    input_text = item["input"]
    analysis = item["output"].get("analysis", "")
    solution = item["output"].get("solution", "")

    mode = random.choice(["diagnosis", "explain", "advice"])

    if mode == "diagnosis":
        return {
            "instruction": INSTRUCTIONS["diagnosis"],
            "input": input_text,
            "output": f"{analysis}，建议：{solution}"
        }

    elif mode == "explain":
        fault_code = item["output"].get("fault_code", "")
        if fault_code:
            return {
                "instruction": INSTRUCTIONS["explain"],
                "input": f"{fault_code}是什么意思？",
                "output": f"{fault_code}表示{analysis}，通常需要{solution}"
            }
        else:
            return None

    elif mode == "advice":
        return {
            "instruction": INSTRUCTIONS["advice"],
            "input": input_text + "，还能继续开吗？",
            "output": f"{analysis}。建议尽快处理：{solution}"
        }


# ===== 多轮生成 =====
def build_multi(item):
    input_text = item["input"]
    analysis = item["output"].get("analysis", "")
    solution = item["output"].get("solution", "")

    # 简单模拟对话
    return {
        "conversations": [
            {"from": "user", "value": input_text},
            {"from": "assistant", "value": "请问是在冷车还是热车时出现？"},
            {"from": "user", "value": "冷车比较明显"},
            {"from": "assistant", "value": f"{analysis}，建议：{solution}"}
        ]
    }


# ===== 主函数 =====
def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    single_data = []
    multi_data = []

    for item in data:
        # ===== 单轮 =====
        single = build_single(item)
        if single:
            single_data.append(single)

        # ===== 多轮（10%概率生成）=====
        if random.random() < 0.1:
            multi = build_multi(item)
            multi_data.append(multi)

    # ===== 保存 =====
    with open(OUTPUT_SINGLE, "w", encoding="utf-8") as f:
        json.dump(single_data, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_MULTI, "w", encoding="utf-8") as f:
        json.dump(multi_data, f, ensure_ascii=False, indent=2)

    print("完成")
    print("单轮数据:", len(single_data))
    print("多轮数据:", len(multi_data))


if __name__ == "__main__":
    main()