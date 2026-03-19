# 项目背景
汽车维修诊断依赖大量维修经验与专业知识，传统故障排查需要查阅维修手册、OBD故障码说明及历史案例，信息来源分散且检索效率较低。
本项目基于 大语言模型（LLM）与RAG技术 构建汽车维修知识助手，通过整合 OBD故障码、维修案例与维修手册数据，实现汽车故障智能诊断与维修建议生成，提升故障排查效率并降低技术人员学习成本。

# 数据预处理
## rag数据
### OBD数据生成
Part of the implementation is adapted from:
https://github.com/CLUEbenchmark/SuperCLUE-Auto.git
测试效果推荐，根据结果选择gbt4-turbo作为数据生成的模型

原始数据
obd code：https://github.com/mytrile/obd-trouble-codes.git
python data_prepare/generate_obd_codes.py
生成数据保存在obd_codes.json

### 汽车维修数据
参考链接：
https://zhuanlan.zhihu.com/p/104690176

查询网上案例100个，再根据大模型扩充维修案例至300个

## 微调指令数据
基于汽车故障知识库，设计数据增强策略，利用大模型进行语义改写与多样化生成，构建1000+高质量指令数据（单轮+多轮），用于大模型微调】
利用rag input解析出 apalaca格式单轮指令集，包含解释类，建议类和诊断论
然后利用chatglm(在汽车售后理解上表达仅次于chatgpt)扩充到1000条，包含950条单轮数据，50条多轮数据
最终生成数据在car_finetune.json

# 方案选型
## finetune
基础模型选型，利用llama_factory自带的eval测评结果，对比一下模型：
Qwen/Qwen2.5-3B-Instruct最均衡，选择此模型
结果如下：
| model | predict_bleu-4 | predict_rouge-l|
|----- | ----- | -----|
| Llama-3.2-3B-Instruct| 2.01 | 9.79 |
| Qwen2.5-3B-Instruct  | 2.83 | 9.50 |
| Qwen2.5-1.5B-Instruct |2.79| 8.87 |
| DeepSeek-R1-Distill-Qwen-1.5B| 3.39 | 7.93|
Qlora 效果  效果显著
| model | predict_bleu-4 | predict_rouge-l|
|----- | ----- | -----|
| Qwen2.5-3B-Instruct-Qlora| 42.72 | 55.38 |

## rag
数据清理 见第一部分
故障码直接查询，如果prompt里边有故障码直接查询
分路召回，避免维修示例把结构知识冲掉或者引用的全是obd code数据过于死板

advance skill使用
prompt rewrite
rerank

最终流程
用户 query
   ↓
🔹 Query Rewrite（规范化问题）
   ↓
🔹 Retrieve（多路召回）
   ↓
🔹 Rerank（重排序）
   ↓
🔹 构建带编号 context
   ↓
🔹 LLM（JSON + 引用）
   ↓
返回：
  answer(JSON) + sources

# 最终部署与效果展示
vllm模板一致性（导出llama_factory模板）
VLLM_USE_MODELSCOPE=true vllm serve /home/bygpu/models/Qwen/Qwen2___5-3B-Instruct --port 7860 --max-model-len 800  --gpu-memory-utilization 0.95 --served-model-name gpt-5-chat

VLLM_USE_MODELSCOPE=true vllm serve /home/bygpu/models/Qwen/Qwen2___5-3B-Instruct-vehicle-qlora --port 7860 --max-model-len 800  --gpu-memory-utilization 0.95 --served-model-name gpt-5-chat --chat-template /home/bygpu/models/Qwen/Qwen2___5-3B-Instruct-vehicle-qlora/chat_template.jinja

demo
后端：fastapi uvicorn api:app --reload --host 127.0.0.1 --port 8000
前端：gradio python gradio_web.py


