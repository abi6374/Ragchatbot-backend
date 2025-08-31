import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.schema import Document

class FaissStore:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # Load free embedding model
        self.embedder = SentenceTransformer(model_name)
        self.dim = self.embedder.get_sentence_embedding_dimension()
        # FAISS index
        self.index = faiss.IndexFlatL2(self.dim)
        self.docs: list[Document] = []

    def add_documents(self, docs: list[Document]):
        texts = [doc.page_content for doc in docs]
        embs = self.embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        self.index.add(embs.astype("float32"))
        self.docs.extend(docs)

    def query(self, query_text: str, top_k: int = 5) -> list[Document]:
        q_emb = self.embedder.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
        distances, indices = self.index.search(q_emb.astype("float32"), top_k)
        return [self.docs[i] for i in indices[0]]
