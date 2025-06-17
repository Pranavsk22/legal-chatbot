import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Initialize Groq client with API key
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load the sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index and associated documents
index = faiss.read_index("retriever/legal_index.faiss")
with open("retriever/legal_docs.pkl", "rb") as f:
    documents = pickle.load(f)


def retrieve_context(query, top_k=3):
    """Retrieve top-k relevant document chunks using semantic similarity."""
    query_embedding = model.encode([query])
    _, indices = index.search(query_embedding, top_k)
    return [documents[i] for i in indices[0]]


def generate_answer(query, context_chunks):
    """Use LLaMA 3 via Groq to generate a legal answer based on context."""
    context = "\n\n".join(context_chunks)
    prompt = (
        "You are a legal assistant. Answer the legal question strictly based on the following legal context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}"
    )

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",  # Switch to llama3-8b-8192 if you want faster/lighter responses
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=512
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error while generating response: {e}"


if __name__ == "__main__":
    query = input("Ask a legal question: ")
    context_chunks = retrieve_context(query)

    print("\n📚 Retrieved Context:\n")
    for i, chunk in enumerate(context_chunks, 1):
        print(f"{i}. {chunk.strip()}\n")

    print("🤖 AI Answer:\n")
    print(generate_answer(query, context_chunks))
