# MiniLawyer-France  
⚖️ **Mini Lawyer** – A lightweight AI chatbot that answers legal questions about French law. Built with **Streamlit** and **FastAPI**, using **RAG (Retrieval-Augmented Generation)** to provide accurate, accessible legal insights.

## Overview  
**MiniLawyer AI** is a Retrieval-Augmented Generation (RAG) chatbot designed to answer legal questions in French law.  
It uses the **Mistral-7B Instruct model** for natural language understanding and reasoning, combined with a **vector database of ~4.3K embeddings** for efficient retrieval of relevant legal texts.  

The project demonstrates how open-source LLMs can be used to build domain-specific assistants that are lightweight, cost-efficient, and easy to deploy.  

## Features  
- ⚡ **Lightweight**: leverages Mistral-7B Instruct with a small vector DB for fast performance.  
- 📚 **Domain-specific**: trained on curated legal texts from **Legifrance** and **Cornell Law**.  
- 🔍 **RAG pipeline**: semantic search retrieves the most relevant passages before generating an answer.  
- 🌐 **Streamlit UI**: clean and interactive web interface.  
- ☁️ **Deployable**: works on Streamlit Community Cloud or any cloud with FastAPI.  
- 🛠️ **REST API**: exposes chatbot capabilities via **FastAPI** for integration with external apps.  

## Tech Stack  
- **LLM**: [Mistral-7B Instruct](https://mistral.ai/)  
- **Embeddings & Vector DB**: ChromaDB (~4,380 embeddings)  
- **Frameworks**: LangChain, Streamlit, FastAPI  
- **Embeddings**: SentenceTransformers (multilingual MiniLM)  
- **Language**: Python  
