# sentenialx/core/semantic_engine/analyzer.py

from sentence_transformers import SentenceTransformer, util

class SemanticAnalyzer:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # lightweight & fast

        # Example attack signatures (semantic vectors)
        self.attack_signatures = {
            "sql_injection": self.model.encode("select * from users where", convert_to_tensor=True),
            "xss": self.model.encode("<script>alert(", convert_to_tensor=True),
            "ssrf": self.model.encode("curl http://internal", convert_to_tensor=True),
        }

    def analyze(self, tokens: list) -> dict:
        joined_text = " ".join(tokens)
        embedding = self.model.encode(joined_text, convert_to_tensor=True)

        threat_scores = {}
        for attack_type, signature in self.attack_signatures.items():
            similarity = util.pytorch_cos_sim(embedding, signature).item()
            threat_scores[attack_type] = round(similarity, 3)

        max_threat = max(threat_scores, key=threat_scores.get)
        return {
            "joined_text": joined_text,
            "threat_scores": threat_scores,
            "likely_threat": max_threat if threat_scores[max_threat] > 0.65 else "benign"
        }