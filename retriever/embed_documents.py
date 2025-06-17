from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Store all chunks here
documents = []

# Read files from raw_acts
for filename in os.listdir("data/raw_acts"):
    with open(f"data/raw_acts/{filename}", "r", encoding="utf-8") as file:
        text = file.read()
        chunks = text.split(". ")  # Naive chunking; can improve later
        documents.extend(chunks)

# Generate embeddings for each chunk
embeddings = model.encode(documents)

# Build a FAISS index
index = faiss.IndexFlatL2(embeddings[0].shape[0])
index.add(embeddings)

# Save the index and docs
faiss.write_index(index, "retriever/legal_index.faiss")
with open("retriever/legal_docs.pkl", "wb") as f:
    pickle.dump(documents, f)

print(f"✅ Embedded {len(documents)} legal text chunks.")
