import re

def rewrite_query(query):
    # 简单规则 + LLM（可扩展）
    if re.search(r'P\d{4}', query):
        return query  # 有故障码直接用

    # 简单增强（你后面可以换LLM）
    return f"汽车故障诊断问题：{query}"