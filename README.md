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


# 方案选型
## finetune
## rag

# 最终部署与效果展示


