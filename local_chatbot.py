"""
local_chatbot.py

✅ Fully Local RAG Chatbot (no API calls)
✅ FAISS Vector Database for fast retrieval
✅ SentenceTransformer local embeddings
✅ Local LLM via Ollama (Mistral, DeepSeek, Llama3)
✅ Flask Web App (Fancy Tourism Frontend)
✅ Bulletproof rebuild and load logic
"""

# ------------------------- #
# 📦 Import Libraries
# ------------------------- #

import os
import pickle
import re
import faiss
import numpy as np
import requests

from flask import Flask, request, jsonify, render_template
import fitz  # PyMuPDF for PDF extraction
from docx import Document
from sentence_transformers import SentenceTransformer

# ------------------------- #
# ⚙️ Config Settings
# ------------------------- #

# FAISS Index and Chunks Save Paths
FAISS_FOLDER = "faiss_index"
FAISS_INDEX_PATH = os.path.join(FAISS_FOLDER, "faiss.index")
CHUNKS_PATH = os.path.join(FAISS_FOLDER, "chunks.pkl")

# Ollama Local Server
OLLAMA_SERVER = "http://localhost:11434"

# Supported Models
SUPPORTED_MODELS = {
    "mistral": "mistral",
    "deepseek": "deepseek-r1:1.5b",
    "llama": "llama3.2:1b"
}

# ------------------------- #
# 📄 Document Extraction Functions
# ------------------------- #

def extract_text_from_pdf(pdf_path):
    """Extract text properly from PDFs using blocks and sorting."""
    doc = fitz.open(pdf_path)
    text_blocks = []
    for page in doc:
        blocks = page.get_text("blocks")
        blocks = sorted(blocks, key=lambda b: (b[1], b[0]))  # sort by vertical position
        for block in blocks:
            if block[4].strip():
                text_blocks.append(block[4].strip())
    return "\n".join(text_blocks)

def extract_text_from_docx(docx_path):
    """Extract text from DOCX."""
    document = Document(docx_path)
    return "\n".join(para.text for para in document.paragraphs)

def load_documents():
    """Load content from all PDFs and Word docs."""
    texts = []

    pdf_files = [
        "D:/vs_code/llm_local_rag_local_model_chatbot/pdf_files/Document1.pdf",
        "D:/vs_code/llm_local_rag_local_model_chatbot/pdf_files/Document2.pdf",
        "D:/vs_code/llm_local_rag_local_model_chatbot/pdf_files/Document3.pdf"
    ]
    docx_files = [
        "D:/vs_code/llm_local_rag_local_model_chatbot/docx_files/Document4.docx",
        "D:/vs_code/llm_local_rag_local_model_chatbot/docx_files/Document5.docx"
    ]

    for file_path in pdf_files:
        if os.path.exists(file_path):
            texts.append(extract_text_from_pdf(file_path))

    for file_path in docx_files:
        if os.path.exists(file_path):
            texts.append(extract_text_from_docx(file_path))

    return "\n".join(texts)

# ------------------------- #
# ✂️ Text Splitting
# ------------------------- #

def split_text(text, chunk_size=500, overlap=100):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# ------------------------- #
# 💾 FAISS Save and Load
# ------------------------- #

def save_faiss_index(index):
    """Save FAISS index to disk."""
    if not os.path.exists(FAISS_FOLDER):
        os.makedirs(FAISS_FOLDER)
    faiss.write_index(index, FAISS_INDEX_PATH)

def load_faiss_index():
    """Load FAISS index from disk."""
    return faiss.read_index(FAISS_INDEX_PATH)

def save_chunks(chunks):
    """Save text chunks."""
    if not os.path.exists(FAISS_FOLDER):
        os.makedirs(FAISS_FOLDER)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

def load_chunks():
    """Load text chunks."""
    with open(CHUNKS_PATH, "rb") as f:
        return pickle.load(f)

# ------------------------- #
# 🧠 Embedding + FAISS Building
# ------------------------- #

def build_faiss_index(chunks, embedding_model):
    """Generate embeddings and build FAISS vector index."""
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

# ------------------------- #
# 🔎 Retrieval
# ------------------------- #

def retrieve_relevant_chunks(query, embedding_model, faiss_index, chunks, top_k=5):
    """Find top-k relevant chunks using FAISS."""
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = faiss_index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

# ------------------------- #
# ✍️ Build the Prompt
# ------------------------- #

def build_prompt(retrieved_chunks, user_query):
    """Build strict hard prompt."""
    context = "\n".join(retrieved_chunks)[:2000]
    prompt = (
        "You must answer using ONLY ONE valid specific English noun or event name.\n"
        "IMPORTANT: Avoid vague city names like 'Sydney'.\n"
        "No explanations. No paragraphs. No thinking aloud.\n"
        "If unsure, just say 'Unknown'.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {user_query}\n"
        "Answer (ONLY one noun or event name):"
    )
    return prompt

# ------------------------- #
# 🤖 Local LLM Query
# ------------------------- #

def query_ollama(prompt, model, debug=False):
    """Send prompt to Ollama and clean up the output."""
    try:
        response = requests.post(
            f"{OLLAMA_SERVER}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "temperature": 0.1,
                "max_tokens": 10,
                "stream": False
            }
        )
        response.raise_for_status()
        raw_response = response.json()["response"].strip()
    except Exception as e:
        print(f"⚠️ Ollama error: {e}")
        return "Unknown"

    # Post-process
    cleaned = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL).strip()
    final_response = cleaned.split()[0] if cleaned else "Unknown"

    junk_words = {"the", "it", "sure", "yes", "maybe"}
    if final_response.lower() in junk_words:
        final_response = "Unknown"

    if debug:
        print("\n=== DEBUG Response ===")
        print(raw_response)
        print("======================\n")

    return final_response

# ------------------------- #
# 🖥️ Flask Web App
# ------------------------- #

app = Flask(__name__)

print("="*60)
print("🚀 Mrig Sydney RAG Chatbot | Local Mode with Ollama")
print("="*60)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Ask user: Rebuild or Load
rebuild = input("🔵 Rebuild FAISS index from scratch? (yes/no): ").strip().lower()

if rebuild == "yes":
    print("🔵 Rebuilding FAISS database...")

    # Delete old saved files if exist
    if os.path.exists(FAISS_INDEX_PATH):
        os.remove(FAISS_INDEX_PATH)
    if os.path.exists(CHUNKS_PATH):
        os.remove(CHUNKS_PATH)

    # Rebuild entire knowledge base
    all_text = load_documents()
    text_chunks = split_text(all_text, chunk_size=500, overlap=100)

    # Manual fact injection (booster knowledge)
    fact_chunks = [
        "The Opal card is Sydney’s public transport smartcard.",
        "The Rocks is the oldest historic area in Sydney.",
        "Vivid Sydney is the biggest arts and music festival.",
        "Surry Hills is famous for local cafes and hidden restaurants.",
        "CBD and Darling Harbour are top choices for first-time visitors."
    ]
    text_chunks += fact_chunks

    faiss_index, _ = build_faiss_index(text_chunks, embedding_model)
    save_faiss_index(faiss_index)
    save_chunks(text_chunks)

    print("✅ FAISS Rebuild complete.")

else:
    print("🔵 Loading existing FAISS index...")
    faiss_index = load_faiss_index()
    text_chunks = load_chunks()

# Model selection
print("🔵 Available Models: mistral | deepseek | llama")
chosen_model_key = input("🔵 Which model to use? (type one): ").strip().lower()
if chosen_model_key not in SUPPORTED_MODELS:
    print(f"⚠️ Invalid model. Defaulting to mistral.")
    chosen_model_key = "mistral"
model_to_use = SUPPORTED_MODELS[chosen_model_key]
print(f"✅ Using model: {model_to_use}")

@app.route("/")
def index():
    """Render front-end UI."""
    return render_template("index.html")

@app.route("/chatbot", methods=["POST"])
def chatbot():
    """Chatbot pipeline."""
    data = request.get_json()
    user_query = data.get("question", "")
    if not user_query:
        return jsonify({"error": "No question provided."}), 400

    try:
        relevant_chunks = retrieve_relevant_chunks(user_query, embedding_model, faiss_index, text_chunks)
        prompt = build_prompt(relevant_chunks, user_query)
        answer = query_ollama(prompt, model=model_to_use)
        return jsonify({"response": answer})
    except Exception as e:
        print(f"⚠️ Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
