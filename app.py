import json
import requests
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

st.set_page_config(page_title="Mini Lawyer", page_icon="⚖️")
st.title("Mini Lawyer - French Law")

# --- Load embeddings & vector DB ---
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
db = Chroma(persist_directory=None, embedding_function=embedding_model)
retriever = db.as_retriever(search_kwargs={"k": 3})

# --- Function to call OpenRouter API ---
def ask_mistral_openrouter(prompt):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": "Bearer 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",
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

# --- Initialize chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": 
                                     "Hello! I’m your Mini Lawyer 🤖. Ask me anything about French law and I’ll explain it clearly."})

# --- Display chat messages ---
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# --- Streamlit input ---
if prompt := st.chat_input("Type your legal question here..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Retrieve relevant documents
    docs = retriever.get_relevant_documents(prompt)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Build prompt for Mistral
    mistral_prompt = f"""
    You are a French law assistant. Answer the user’s question **using ONLY the information in the documents provided below**.
    If the answer cannot be found in the documents, reply: "I could not find the answer in the sources."

    Documents:
    {context}

    Question:
    {prompt}

    Answer concisely and clearly.
    """

    # Call Mistral
    answer = ask_mistral_openrouter(mistral_prompt)

    # Add assistant message to chat
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)

    # Optionally, show sources in a collapsible section
    with st.expander("📚 Sources"):
        for doc in docs:
            st.write("-", doc.metadata.get("source", "Unknown"))
