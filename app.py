import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from processor import DocumentProcessor
from model_store import FaissStore
from rag_engine import RAGEngine
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
processor = DocumentProcessor()
store = FaissStore()
rag = RAGEngine(store)

@app.get("/health")
async def health_check():
    return {"status": "Running properly"}

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files supported")
    content = await file.read()
    docs = processor.process_pdf(content)
    store.add_documents(docs)
    return {"status": "indexed", "chunks": len(docs)}

class QueryRequest(BaseModel):
    question: str
    provider: str = "groq"  # default provider

@app.post("/query")
async def post_query(query_request: QueryRequest):
    if store.index.ntotal == 0:
        raise HTTPException(status_code=400, detail="No documents indexed yet")
    result = rag.answer(query_request.question, provider=query_request.provider)
    return {
        "answer": result["answer"]
        # "sources": [
        #     {"content": doc.page_content, "metadata": doc.metadata}
        #     for doc in result["sources"]
        # ],
    }

# @app.get("/query")
# async def query(
#     question: str = Query(..., min_length=1),
#     provider: str = Query("groq", regex="^(groq|gemini)$")
# ):
#     if store.index.ntotal == 0:
#         raise HTTPException(status_code=400, detail="No documents indexed yet")
#     result = rag.answer(question, provider=provider)
#     return {
#         "answer": result["answer"],
#         "sources": [
#             {"content": doc.page_content, "metadata": doc.metadata}
#             for doc in result["sources"]
#         ],
#     }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
