import os
import streamlit as st
from dotenv import load_dotenv

# Must be the first Streamlit command
st.set_page_config(page_title="⚖️ Indian Legal Chatbot", layout="centered")
st.title("🧑‍⚖️ AI Legal Assistant (India)")
# ENV loading
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Optional debug: confirm key
st.write("🔑 GROQ key loaded:", GROQ_API_KEY[:6] + "…" if GROQ_API_KEY else "❌ MISSING")
# ───────────────────────────────
# ───────────────────────────────
# Lazy‑load heavy stuff
# ───────────────────────────────
#@st.cache_resource(show_spinner=False)
def load_retriever():
    from sentence_transformers import SentenceTransformer
    import faiss, pickle

    #model = SentenceTransformer("all-MiniLM-L6-v2")
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    index = faiss.read_index("retriever/legal_index.faiss")
    with open("retriever/legal_docs.pkl", "rb") as f:
        docs = pickle.load(f)

    return model, index, docs


@st.cache_resource(show_spinner=False)
def load_groq():
    from groq import Groq
    return Groq(api_key=GROQ_API_KEY)


# ───────────────────────────────
# 4. Helper functions
# ───────────────────────────────
def retrieve_context(model, index, docs, query, k=3):
    emb = model.encode([query])
    _, idxs = index.search(emb, k)
    return [docs[i] for i in idxs[0]]


def generate_answer(client, context_chunks, question):
    ctx = "\n\n".join(context_chunks)
    prompt = (
        "You are a legal assistant. Answer the Indian legal question "
        "using ONLY the context below.\n\n"
        f"Context:\n{ctx}\n\nQuestion: {question}"
    )

    resp = client.chat.completions.create(
        model="llama3-8b-8192",     # small & fast
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=512,
    )
    return resp.choices[0].message.content.strip()


# ───────────────────────────────
# 5. UI
# ───────────────────────────────
query = st.text_input("Enter your legal question:")

if query:
    st.info("⚙️ Model loading the first time may take ~10 sec…")
    with st.spinner("Thinking like a lawyer…"):
        model, index, docs = load_retriever()
        client = load_groq()

        chunks = retrieve_context(model, index, docs, query)
        answer = generate_answer(client, chunks, query)

    # ───── results ─────
    st.subheader("💬 AI Answer")
    st.write(answer)

    st.subheader("📚 Retrieved Context")
    for i, ch in enumerate(chunks, 1):
        st.markdown(f"**{i}.** {ch.strip()}")

    st.caption("⚠️ This is informational only and **not** legal advice.")
