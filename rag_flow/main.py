from llama_index.core import Settings
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager

from rag_flow.build_index import build_index
from rag_flow.llms import initialize_llm, initialize_embedding
from rag_flow.retriever import retrieve


obd_path = r"C:\Users\qian gao\git_project\AutoDiagGPT\data_prepare\rag_data\obd_codes.json"
case_path = r"C:\Users\qian gao\git_project\AutoDiagGPT\data_prepare\rag_data\final_dataset.json"
vector_path="./chroma_db"
embed_model = initialize_embedding("bge")
llm = initialize_llm("qwen")
index  = build_index(obd_path, case_path, persist_dir=vector_path, embed_model=embed_model)

def generate_answer(query, nodes):
    llm = Settings.llm

    # 👉 拼 context
    context = "\n\n".join([n.text for n in nodes])

    prompt = f"""
你是汽车故障诊断专家，请根据以下信息回答：

{context}

用户问题：{query}

请输出：
1. 可能原因
2. 检查建议
3. 维修方案
"""

    resp = llm.complete(prompt) 
    answer = resp.text

    # 👉 构造来源（重点）
    sources = []
    for n in nodes:
        sources.append({
            "text": n.text[:200],   # 截断一下
            "score": getattr(n, "score", None),
            "metadata": n.metadata
        })

    return answer, sources

def main(query):
    # 初始化
    Settings.llm = llm
    Settings.embed_model = embed_model
   
    #  使用LlamaDebugHandler构建事件回溯器，以追踪LlamaIndex执行过程中发生的事件
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])
    Settings.callback_manager = callback_manager

    nodes = retrieve(index, query)
   
    answer, sources = generate_answer(query, nodes)
    return answer, sources


if __name__ == "__main__":

    #  查询测试
    query = "我的车报了P0173故障码，怎么办？"
    answer, sources = main(query)
    print("诊断结果:", answer)
    print("相关来源:", sources)
    