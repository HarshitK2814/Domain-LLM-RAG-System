from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

documents = [
    "Supervised learning uses labeled data.",
    "Overfitting happens when model memorizes training data."
]

embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(documents)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

def retrieve(query):
    query_vec = embedder.encode([query])
    D, I = index.search(np.array(query_vec), k=1)
    return documents[I[0][0]]

if __name__ == "__main__":
    print(retrieve("What is supervised learning?"))
