from sentence_transformers import SentenceTransformer
import numpy as np
import json

model = SentenceTransformer('all-MiniLM-L6-v2')

# Dummy classification threshold and label mapping
threshold = 0.85
known_threats = {
    "prompt injection": "Avoid vague instructions like 'ignore above', 'pretend to be', etc.",
    "sql injection": "Usage of ' OR 1=1 -- style payloads",
}

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def analyze_with_embeddings(text):
    embedding = model.encode(text)
    similarities = {}
    for label, example in known_threats.items():
        example_emb = model.encode(example)
        score = cosine_similarity(embedding, example_emb)
        similarities[label] = score

    highest = max(similarities.items(), key=lambda x: x[1])
    if highest[1] > threshold:
        return {"threat": highest[0], "score": highest[1]}
    return {"threat": "none", "score": highest[1]}

