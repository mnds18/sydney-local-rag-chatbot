![Lint Status](https://github.com/mnds18/sydney-local-rag-chatbot/actions/workflows/python-lint.yml/badge.svg)


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


🔑 Environment Variables
Variable	Purpose
OLLAMA_HOST	Ollama server URL (default: http://localhost:11434)
Ensure Ollama is installed locally with required models downloaded!

🖥️ Demo Screenshots

🏗️ Setup Instructions

# 1. Clone repo
git clone https://github.com/yourusername/sydney-local-rag-chatbot.git
cd sydney-local-rag-chatbot

# 2. (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate  # (Linux/Mac)
venv\Scripts\activate     # (Windows)

# 3. Install required packages
pip install -r requirements.txt

# 4. Run the app
python local_chatbot.py


Access chatbot at: http://127.0.0.1:5000

🛡️ License
This project is licensed under the MIT License.

🤝 Contributing
Contributions are welcome! 🚀

Please create an issue before submitting pull requests.
Ensure you follow basic Python code style guidelines (flake8).

📢 Acknowledgments
FAISS (Facebook AI Similarity Search)

Sentence Transformers (HuggingFace)

Ollama - Local LLM Runtime

Flask Framework

🔥 Connect
If you're passionate about the intersection of
Data Science × AI Engineering × Product Thinking — let's connect!
Always building smart, offline-first AI products! 🚀

LinkedIn : https://www.linkedin.com/in/mrigendranath/

