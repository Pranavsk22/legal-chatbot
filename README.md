# legal-chatbot
AI based Legal Chatbot


This is an AI-based Legal chatbot that :
1. Implements a **Retrieval-Augmented Generation (RAG)** architecture.
2. Trains the system on **Indian legal codes**, **precedents**, and **regulations**.
3. Ensures the chatbot understands **legal terminology** and **context**.
4. Develops a **user-friendly interface** for client interaction.
5. Implements **safeguards** to prevent incorrect legal advice.
6. Includes **disclaimers** about AI legal guidance limitations.

# ⚖️ Indian Legal Chatbot • RAG‑Powered AI Assistant

![Streamlit](https://img.shields.io/badge/Built&nbsp;with-Streamlit-ff4b4b?logo=streamlit&logoColor=white)
![License](https://img.shields.io/github/license/Pranavsk22/legal-chatbot?color=blue)
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Status](https://img.shields.io/badge/Status-Alpha-yellow)

Retrieval‑Augmented Generation (RAG) chatbot that answers **Indian‑law questions** with citations to the Indian Penal Code and related sections.  
Runs a fast MiniLM embedder + FAISS index locally and calls **Groq’s Llama‑3** models for answer generation.

> **Disclaimer**  
> This project is for educational use only. It does **not** constitute legal advice. Always consult a qualified lawyer for real‑world matters.

---

## ✨ Key Features
| Feature | Details |
|---------|---------|
| 🔍 **Semantic search** | Sentence‑Transformers (`paraphrase‑MiniLM‑L6‑v2`) + FAISS (`IndexFlatL2`) |
| 📚 **Context‑aware RAG** | Retrieves top‑**k** code sections, feeds them to Llama‑3‑8B via Groq API |
| 🖥️ **Streamlit UI** | Clean single‑page chat with context disclosure |
| ☁️ **One‑click deploy** | Works on Streamlit Community Cloud / HF Spaces / Render |
| 🛡️ **Safety guardrails** | Model instructed *not* to guess; always cites sections; reminder banner |

---

## 🗂️ Project Structure
legal-chatbot/
├─ app.py ← Streamlit front‑end + RAG pipeline
├─ requirements.txt
├─ .env.example ← template for your API key
├─ retriever/
│ ├─ embed_documents.py ← script to build FAISS index
│ ├─ legal_index.faiss ← vector index (~1.7 MB)
│ └─ legal_docs.pkl ← original text chunks (~150 KB)
└─ ...


---

## 🚀 Quick Start

### 1  Clone & install
git clone https://github.com/Pranavsk22/legal-chatbot.git
cd legal-chatbot
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

2  Add your Groq key
cp .env.example .env        # then edit:
# .env
GROQ_API_KEY=gsk_your_key_here

3  Run locally
streamlit run app.py
Open the URL shown in the terminal → ask “What is the punishment for theft?”

🌐 Deploy to Streamlit Cloud
Push this repo to GitHub.

Go to https://share.streamlit.io → New app.

Point to your repo/branch, add Secrets:
GROQ_API_KEY="gsk_your_key_here"
Deploy.
Cold start ≈ 15 s while the FAISS index & model load.

🔧 Regenerating the Vector Index
If you add more statutes / judgments:


python retriever/embed_documents.py --src data/extra_laws.txt \
                                    --dest retriever/legal_index.faiss
git add retriever/legal_index.faiss retriever/legal_docs.pkl
git commit -m "Re‑embed corpus"
git push
(The script chunks text to 512 chars, embeds with MiniLM, and writes both index & pickle.)

⚙️ Key Dependencies
Package	Role
streamlit 1.34	UI
groq 0.29	Chat completion (Llama‑3)
sentence-transformers 1.2.1	MiniLM embeddings
faiss‑cpu 1.7.4	Vector similarity search
torch 1.13.1	Backend for MiniLM

🤝 Contributing
Pull requests are welcome — especially:

Adding more Indian legal corpora (Case‑Law, CPC, CrPC)

UI/UX polish (chat history, context toggle)

Unit tests for embed/retrieval pipeline

📜 License
MIT © 2025 Pranav
Feel free to fork, learn, and build your own legal copilots!

