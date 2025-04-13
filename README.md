![Lint Status](https://github.com/mrig18/sydney-local-rag-chatbot/actions/workflows/python-lint.yml/badge.svg)


<p align="center">
  <img src="https://img.shields.io/badge/Powered%20By-LangChain-blue" />
  <img src="https://img.shields.io/badge/Vectorstore-FAISS-green" />
  <img src="https://img.shields.io/badge/Language-Python-yellow" />
  <img src="https://img.shields.io/badge/Model-Ollama%20(Local%20LLM)-orange" />
</p>

<br />

# 🐨 Mrig's Sydney Local RAG Chatbot

An end-to-end **Retrieval-Augmented Generation (RAG)** chatbot built **entirely locally** — no APIs, no cloud services.  
It reads **PDF** and **Word documents**, builds a **vectorstore**, retrieves answers using **FAISS**, and **generates responses using a local LLM** (Mistral / DeepSeek / Llama3 running in Ollama).

<br />

---

## 🚀 Features

- ✅ Full **local-only** document understanding (PDFs + DOCXs)
- ✅ **FAISS** vector database for fast document search
- ✅ **Hard prompting** forcing 1-word or short answers
- ✅ Supports multiple **local LLMs** (DeepSeek, Mistral, Llama3 via Ollama)
- ✅ **Beautiful tourism-themed** front-end chatbot (Flask)
- ✅ Fully **rebuildable vectorstore** on demand
- ✅ Clean, highly commented, production-quality code
- ✅ Designed for **easy personal deployment** or **professional demos**

<br />

---

## 📂 Project Structure

```plaintext
.
├── local_chatbot.py      # Main backend: extraction + FAISS + Ollama + Flask
├── templates/
│   └── index.html        # Frontend chat UI
├── faiss_index/
│   ├── faiss.index       # Saved FAISS vector database
│   └── chunks.pkl        # Saved split document chunks
├── pdf_files/            # Your source PDFs
├── docx_files/           # Your source DOCX files
├── README.md             # You are here :)
