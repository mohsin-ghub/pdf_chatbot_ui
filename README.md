# 🤖 PDF Chatbot with TinyLlama & Streamlit

A simple chatbot app that answers questions from any uploaded PDF using a local TinyLlama model and ChromaDB for retrieval.

---

## 🧩 Features

- 📂 Upload your own PDF or drag and drop in UI
- 💬 Ask questions about it
- 🧠 Uses TinyLlama 1.1B (locally)
- 🎛️ Streamlit UI

---

## 🚀 Setup

### 1. Install dependencies

pip install streamlit PyPDF2 transformers sentence-transformers chromadb

2. Download TinyLlama model

Visit: https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0


Place it in the project folder

▶️ Run

streamlit run app.py

📁 Project Files

app.py – main Streamlit app

tinyllama-1.1b-chat-v1.0.Q2_K.gguf – (downloaded model)
