#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import os
from collections import defaultdict

# 读取OBD code数据
obd_file = r"third_party\obd-trouble-codes\obd-trouble-codes.csv"
obd_codes = []

with open(obd_file, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) >= 2:
            code = row[0].strip('"')
            description = row[1].strip('"')
            obd_codes.append({'code': code, 'description': description})

print(f"总共读取 {len(obd_codes)} 条OBD code")

# 按照OBD code的分类来排序，选择最常用的
# P codes (Powertrain) 最常用，优先级最高
# B codes (Body)
# C codes (Chassis)
# N codes (Network)
# U codes (User defined)

def get_code_prefix(code):
    """获取code的前缀，用于分类"""
    if code:
        return code[0]
    return ""

def code_sort_key(code_str):
    """为排序创建key，使得P codes优先，然后按字母数字排序"""
    prefix = code_str[0] if code_str else ""
    # 顺序：P, B, C, N, U （从最常用到最少用）
    prefix_order = {'P': 0, 'B': 1, 'C': 2, 'N': 3, 'U': 4}
    prefix_priority = prefix_order.get(prefix, 5)
    # 数字部分用于第二级排序
    try:
        num = int(code_str[1:]) if len(code_str) > 1 else 0
    except:
        num = 0
    return (prefix_priority, num)

# 按优先级排序
obd_codes_sorted = sorted(obd_codes, key=lambda x: code_sort_key(x['code']))

# 取前300条最常用的
top_300 = obd_codes_sorted[:300]

print(f"选取前300条最常用OBD code")

# 统计各类型的数量
type_count = defaultdict(int)
for item in top_300:
    prefix = get_code_prefix(item['code'])
    type_count[prefix] += 1

print("\n各类型OBD code统计:")
for prefix in sorted(type_count.keys()):
    print(f"  {prefix} codes: {type_count[prefix]} 条")

# 保存为CSV
output_dir = "data_prepare"
output_file = os.path.join(output_dir, "top_300_obd_codes.csv")

os.makedirs(output_dir, exist_ok=True)

with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['code', 'description'])
    writer.writeheader()
    writer.writerows(top_300)

print(f"\n成功保存 {len(top_300)} 条OBD code到: {output_file}")
