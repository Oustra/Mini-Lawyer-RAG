# MiniLawyer-France
⚖️ Mini Lawyer – A lightweight AI chatbot that answers legal questions about French law. Built with Streamlit, using RAG (Retrieval-Augmented Generation) to provide accurate, accessible legal insights.

## Overview  
MiniLawyer AI is a Retrieval-Augmented Generation (RAG) chatbot designed to answer legal questions for a specific jurisdiction.  
It uses the **Mistral-7B Instruct model** for natural language understanding and reasoning, combined with a **vector database of ~4.3K embeddings** for efficient retrieval of relevant legal texts.  

This project demonstrates how open-source LLMs can be used to build domain-specific assistants that are lightweight, cost-efficient, and easy to deploy.  

## Features  
- ⚡ **Lightweight**: runs on Mistral-7B Instruct + small vector DB.  
- 📚 **Domain-specific**: trained on curated legal texts (e.g., Legifrance / Cornell Law).  
- 🔍 **RAG pipeline**: semantic search retrieves the most relevant passages before generating an answer.  
- 🌐 **Streamlit app**: clean and interactive UI.  
- ☁️ **Deployable on Streamlit Community Cloud**.  

## Tech Stack  
- **LLM**: [Mistral-7B Instruct](https://mistral.ai/)  
- **Embeddings & Vector DB**: ChromaDB (4,380 embeddings)  
- **Frameworks**: LangChain, Streamlit  
- **Language**: Python  
