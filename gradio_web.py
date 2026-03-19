# gradio_app.py
import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000/chat"

def chat_fn(message):
    resp = requests.post(API_URL, json={"query": message}).json()
    
    answer = resp["answer"]
    sources = resp["sources"]

    # 格式化来源显示
    source_text = ""
    for i, s in enumerate(sources):
        source_text += f"""
### 来源 {i+1}
- 类型: {s['metadata'].get('type', '未知')}
- 内容: {s['text']}
- 相关分数: {s.get('score')}
"""

    return answer, source_text

with gr.Blocks() as demo:
    gr.Markdown("## 🚗 汽车故障诊断系统")
    
    with gr.Row():
        query_input = gr.Textbox(label="输入问题或故障码", placeholder="例如：我的车报了P1073故障码")
        submit_btn = gr.Button("查询")

    with gr.Row():
        answer_box = gr.Textbox(label="诊断结果")
    
    with gr.Accordion("📚 查看参考来源", open=False):
        sources_box = gr.Markdown()

    submit_btn.click(chat_fn, inputs=query_input, outputs=[answer_box, sources_box])

demo.launch()