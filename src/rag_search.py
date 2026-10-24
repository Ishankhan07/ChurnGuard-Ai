# Step 7: Load Saved Knowledge Base

import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


index = faiss.read_index("knowledge_base/faiss_index.bin")

with open("knowledge_base/chunks.pkl", "rb") as file:
    chunks = pickle.load(file)

model = SentenceTransformer("all-MiniLM-L6-v2")


# Search Knowledge Base

def search_knowledge(query, k=3):

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding, dtype="float32")

    distances, indices = index.search(query_embedding, k)

    results = []

    for i in indices[0]:
        if i != -1:
            results.append(chunks[i])

    return results


def answer_question(query):

    results = search_knowledge(query, k=3)

    context = ""

    for result in results:
        context += "\nSource: " + result["source"]
        context += "\n" + result["text"] + "\n"

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": "Answer only from the provided context. If information is missing, say so. Cite sources using their PDF filenames."
            },
            {
                "role": "user",
                "content": "Question: " + query + "\nContext:\n" + context
            }
        ]
    )

    return response.choices[0].message.content


print(answer_question("What are the main reasons for customer churn?"))
