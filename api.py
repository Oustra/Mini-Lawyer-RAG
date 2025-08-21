from fastapi import FastAPI
from pydantic import BaseModel
import os
import json
import requests
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Load environment variables
load_dotenv(dotenv_path="api.env")
api_key = os.getenv("OPENROUTER_API_KEY")

# Initialize FastAPI
app = FastAPI(title="Mini Lawyer API", version="1.0")

# --- Load embeddings & vector DB ---
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
db = Chroma(persist_directory="chroma_db", embedding_function=embedding_model)
retriever = db.as_retriever(search_kwargs={"k": 3})

# --- OpenRouter call function ---
def ask_mistral_openrouter(prompt: str) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistralai/mistral-7b-instruct:free",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.2
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error {response.status_code}: {response.text}"

# --- Request schema ---
class Question(BaseModel):
    query: str

# --- Endpoint ---
@app.post("/ask")
def ask_question(question: Question):
    docs = retriever.get_relevant_documents(question.query)
    context = "\n\n".join([doc.page_content for doc in docs])

    mistral_prompt = f"""
    You are a French law assistant. 
    Answer the user’s question using ONLY the documents provided below.
    - If the documents are in a different language than the question, translate them into the question's language before answering.
    - If the answer cannot be found in the documents, respond: "I could not find the answer in the sources."
    - Keep your answer concise, clear, and professional.

    Documents:
    {context}

    Question:
    {question.query}

    Answer:
    """

    answer = ask_mistral_openrouter(mistral_prompt)

    return {
        "question": question.query,
        "answer": answer,
        "sources": [doc.metadata.get("source", "Unknown") for doc in docs]
    }
