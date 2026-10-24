
from pypdf import PdfReader
import os

folder = "knowledge_base"

documents = []

for file in os.listdir(folder):
    if file.endswith(".pdf"):
        reader = PdfReader(os.path.join(folder, file))

        text = ""

        for page in reader.pages:
            text += page.extract_text() or ""

        documents.append({
            "source": file,
            "text": text
        })

print("PDF files loaded:", len(documents))

for doc in documents:
    print(doc["source"], "Characters:", len(doc["text"]))



# Step 2: Text chunking

chunks = []

chunk_size = 1000
overlap = 200

for doc in documents:
    text = doc["text"]

    for i in range(0, len(text), chunk_size - overlap):
        chunk_text = text[i:i + chunk_size]

        if chunk_text.strip():
            chunks.append({
                "text": chunk_text,
                "source": doc["source"]
            })

print("Total chunks:", len(chunks))
print("\nFirst chunk:")
print(chunks[0]["text"])
print("\nSource:", chunks[0]["source"])




# Step 3: Create embeddings

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = []

for chunk in chunks:
    texts.append(chunk["text"])

embeddings = model.encode(texts, show_progress_bar=True)

print("Embeddings shape:", embeddings.shape)



# Step 4: Store embeddings in FAISS

import faiss
import numpy as np

embeddings = np.array(embeddings, dtype="float32")

index = faiss.IndexFlatL2(embeddings.shape[1])

index.add(embeddings)

print("Total vectors stored:", index.ntotal)




# Step 5: Search Knowledge Base

def search_knowledge(query, k=3):

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding, dtype="float32")

    distances, indices = index.search(query_embedding, k)

    results = []

    for i in indices[0]:
        if i != -1:
            results.append(chunks[i])

    return results


# Test Search

results = search_knowledge(
    "What are the main reasons for customer churn?"
)

for result in results:
    print("\nSource:", result["source"])
    print(result["text"][:300])



# Step 6: Save FAISS Index and Chunks

import pickle

faiss.write_index(index, "knowledge_base/faiss_index.bin")

with open("knowledge_base/chunks.pkl", "wb") as file:
    pickle.dump(chunks, file)

print("Knowledge base saved successfully!")