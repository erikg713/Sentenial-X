sentenial_core/cortex/malicious_embedding_analyzer.py

import numpy as np import torch from transformers import AutoTokenizer, AutoModel

class MaliciousEmbeddingAnalyzer: def init(self, model_name='sentence-transformers/paraphrase-mpnet-base-v2'): self.tokenizer = AutoTokenizer.from_pretrained(model_name) self.model = AutoModel.from_pretrained(model_name)

def embed_payload(self, payload: str) -> np.ndarray:
    inputs = self.tokenizer(payload, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = self.model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

def classify_embedding(self, embedding: np.ndarray, threat_vectors: dict, threshold: float = 0.8) -> str:
    max_score = 0
    best_match = "benign"
    for label, vec in threat_vectors.items():
        score = self._cosine_similarity(embedding, vec)
        if score > threshold and score > max_score:
            max_score = score
            best_match = label
    return best_match

def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

Example Threat Vectors (static or updated dynamically by Cortex)

threat_vectors_example = { "sql_injection": np.random.rand(768), "xss": np.random.rand(768), "os_command": np.random.rand(768) }

Usage

analyzer = MaliciousEmbeddingAnalyzer()

embedding = analyzer.embed_payload("' OR 1=1 --")

label = analyzer.classify_embedding(embedding, threat_vectors_example)

print("Threat label:", label)


