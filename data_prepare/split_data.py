import json
import random

INPUT_FILE = r"C:\Users\qian gao\git_project\AutoDiagGPT\data_prepare\instruction_data\car_finetune.json"
TEST_FILE = "test_100.json"

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ===== 随机抽100条（允许重复）=====
    test_data = random.sample(data, 100)

    with open(TEST_FILE, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print("测试集生成完成:", len(test_data))

if __name__ == "__main__":
    main()