import json
import os
import chromadb
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def obd_to_text(item):
    return f"""
故障码: {item['code']}
描述: {item['description']}
症状: {', '.join(item['symptoms'])}
可能原因: {', '.join(item['possible_causes'])}
解决方案: {', '.join(item['solutions'])}
"""


def case_to_text(item):
    return f"""
用户问题: {item['input']}
分析: {item.get('output', {}).get('analysis', '无分析')}
解决方案: {item['output']['solution']}
是否设计故障码: {item['output']['fault_code']}
"""


def build_index(obd_path, case_path, persist_dir="./chroma_db", embed_model=None):
    with open(obd_path, 'r', encoding='utf-8') as f:
        obd_data = json.load(f)

    with open(case_path, 'r', encoding='utf-8') as f:
        case_data = json.load(f)

    documents = []

    for item in obd_data:
        documents.append(Document(
            text=obd_to_text(item),
            metadata={"type": "obd", "code": item['code']}
        ))

    for item in case_data:
        documents.append(Document(
            text=case_to_text(item),
            metadata={"type": "case"}
        ))

    

    chroma_client = chromadb.PersistentClient(
        path=persist_dir  # 本地目录
    )
    collection = chroma_client.get_or_create_collection("car_rag")

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    
    # 👇 判断是否已有数据
    if collection.count() > 0:
        print("✅ Load existing index")
        return VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model
        )
    else:
        print("⚠️ Build new index")
        return VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=embed_model
        )
