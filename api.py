# api.py
from fastapi import FastAPI
from pydantic import BaseModel
from rag_flow.main import main 

app = FastAPI(title="AutoDiagGPT RAG API")

class QueryRequest(BaseModel):
    query: str

@app.post("/chat")
def chat(req: QueryRequest):
    """
    输入 query，返回诊断结果 + 来源
    """
    answer, sources = main(req.query)

    # 返回 JSON
    return {
        "answer": answer,
        "sources": sources
    }