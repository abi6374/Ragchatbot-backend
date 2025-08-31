import os
import requests
import google.generativeai as genai
from groq import Groq
from langchain.schema import Document

# Configure Gemini
genai.configure(api_key="AIzaSyBsFdZIzGNlvzd-caz-_hANbBtX2JfPOOQ")

# Initialize Groq
groq_client = Groq(api_key="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")

class RAGEngine:
    def __init__(self, store):
        self.store = store

    def _call_groq(self, prompt, model="llama-3.1-8b-instant"):
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.1,
            max_tokens=1024
        )
        return response.choices[0].message.content

    def _call_gemini(self, prompt, model="gemini-1.5-flash"):
        response = genai.GenerativeModel(model).generate_content(prompt)
        return response.text

    def answer(self, question: str, provider="groq") -> dict:
        # Retrieve relevant chunks
        docs: list[Document] = self.store.query(question)
        context = "\n---\n".join([doc.page_content for doc in docs])

        # Build prompt
        prompt = (
            "You are an expert legal document analyst. Use ONLY the provided context to answer the question.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )

        # Generate answer
        if provider == "groq":
            try:
                answer = self._call_groq(prompt)
            except Exception:
                answer = self._call_gemini(prompt)
        else:
            answer = self._call_gemini(prompt)

        return {"answer": answer, "sources": docs}
