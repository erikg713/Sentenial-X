
# sentenial_core/compliance/regulatory_vector_matcher.py

"""
Regulatory Vector Matcher
-------------------------
Matches parsed legal ontology vectors against internal system controls
and compliance requirements using semantic similarity and vector embeddings.

This module bridges regulatory mandates to system controls via dense embeddings,
enabling precise compliance gap analysis and automated remediation suggestions.
"""

from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer, util
import logging

logger = logging.getLogger(__name__)

class RegulatoryVectorMatcher:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        # Load efficient sentence transformer model for embeddings
        self.model = SentenceTransformer(embedding_model_name)
        # Preload internal control vectors (mocked here, replace with real data)
        self.internal_controls = {
            "Data Encryption": "All personal data must be encrypted at rest and in transit.",
            "Access Control": "Only authorized personnel can access sensitive data.",
            "Incident Reporting": "Security incidents must be reported within 72 hours.",
        }
        self.control_embeddings = self.model.encode(list(self.internal_controls.values()), convert_to_tensor=True)
        logger.info("RegulatoryVectorMatcher initialized with internal control embeddings")

    def match_regulations_to_controls(self, regulations: List[str], threshold: float = 0.7) -> Dict[str, List[Tuple[str, float]]]:
        """
        Matches a list of regulation clauses to internal controls by semantic similarity.

        Args:
            regulations (List[str]): Regulatory clauses or obligations.
            threshold (float): Similarity threshold for matching.

        Returns:
            Dict[str, List[Tuple[str, float]]]: Mapping of regulation -> list of (control, similarity).
        """
        results = {}
        reg_embeddings = self.model.encode(regulations, convert_to_tensor=True)
        for reg, reg_emb in zip(regulations, reg_embeddings):
            hits = []
            # Compute cosine similarities
            cos_scores = util.cos_sim(reg_emb, self.control_embeddings)[0]
            for idx, score in enumerate(cos_scores):
                sim = float(score)
                if sim >= threshold:
                    hits.append((list(self.internal_controls.keys())[idx], sim))
            hits.sort(key=lambda x: x[1], reverse=True)
            results[reg] = hits
            logger.debug(f"Regulation matched: {reg} -> {hits}")
        return results

    def add_internal_control(self, name: str, description: str):
        """
        Add a new internal control with its description to the matcher.

        Args:
            name (str): Control name.
            description (str): Control description.
        """
        self.internal_controls[name] = description
        # Update embeddings with new control
        descriptions = list(self.internal_controls.values())
        self.control_embeddings = self.model.encode(descriptions, convert_to_tensor=True)
        logger.info(f"Added new internal control: {name}")

# Example usage for unit test or CLI tool
if __name__ == "__main__":
    matcher = RegulatoryVectorMatcher()
    regulations_sample = [
        "Organizations must encrypt all personal data during storage and transmission.",
        "Data breach incidents must be reported promptly to authorities.",
    ]
    matches = matcher.match_regulations_to_controls(regulations_sample)
    for reg, hits in matches.items():
        print(f"Regulation: {reg}\nMatches: {hits}\n")
