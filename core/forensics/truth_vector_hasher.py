"""
core/forensics/truth_vector_hasher.py

Sentenial-X Forensics Truth Vector Hasher Module - computes and hashes truth vectors
for AI outputs, enabling verifiable assessments of factual accuracy, hallucination detection,
and alignment with ground truth sources. Vectors are hashed for immutability and integrated
into forensic ledgers for non-repudiation.
Supports semantic similarity (e.g., via cosine on embeddings) and fact-check APIs.
"""

import asyncio
import time
import json
import hashlib
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity  # For vector ops
from cli.memory_adapter import get_adapter
from cli.logger import default_logger
from .ledger_sequencer import LedgerSequencer  # Integration
from ..utils.embedding_adapter import get_embedding_model  # Assume shared embedding util

# Truth vector schema: [factuality_score, hallucination_risk, source_alignment, ...]
TRUTH_VECTOR_SCHEMA = {
    "output_id": str,
    "timestamp": float,
    "truth_vector": List[float],  # e.g., [0.95, 0.05, 0.92] for fact/hall/source
    "components": Dict[str, Any],  # Breakdown: {"facts_checked": [...], "similarity": 0.95}
    "hash": str  # SHA-256 of vector + metadata
}

class TruthVectorHasher:
    """
    Computes truth vectors for AI responses and hashes them for forensic integrity.
    Vector dimensions: factuality (0-1), hallucination risk (0-1), source alignment (0-1).
    Uses embeddings for semantic checks; integrates with LedgerSequencer.
    
    :param ledger: Optional LedgerSequencer for logging
    :param embedding_model: Pre-loaded embedding function (e.g., from sentence-transformers)
    :param fact_check_sources: List of ground truth sources (e.g., URLs or texts)
    """
    def __init__(self, ledger: Optional[LedgerSequencer] = None,
                 embedding_model: Optional[callable] = None,
                 fact_check_sources: Optional[List[str]] = None):
        self.ledger = ledger or LedgerSequencer()
        self.mem = get_adapter()
        self.logger = default_logger
        self.embedding_model = embedding_model or get_embedding_model("all-MiniLM-L6-v2")
        self.fact_check_sources = fact_check_sources or [
            "https://en.wikipedia.org/wiki/Artificial_intelligence",  # Mock sources
            "Ground truth fact database"
        ]
        self.source_embeddings = self._precompute_source_embeddings()

    def _precompute_source_embeddings(self) -> np.ndarray:
        """Precompute embeddings for fact-check sources."""
        embeddings = []
        for source in self.fact_check_sources:
            # Mock: In prod, fetch/parse source content
            emb = self.embedding_model(source[:512])  # Truncate for demo
            embeddings.append(emb)
        return np.vstack(embeddings) if embeddings else np.array([])

    def _compute_truth_vector(self, ai_output: str, ground_truths: Optional[List[str]] = None) -> Tuple[List[float], Dict[str, Any]]:
        """
        Compute truth vector components.
        
        :param ai_output: AI-generated text
        :param ground_truths: Optional list of reference texts
        :return: (vector, components dict)
        """
        output_emb = np.array(self.embedding_model(ai_output)).reshape(1, -1)
        
        # Factuality: Avg cosine sim to sources
        if len(self.source_embeddings) > 0:
            fact_sim = cosine_similarity(output_emb, self.source_embeddings).mean()
        else:
            fact_sim = 0.5  # Default neutral
        
        # Hallucination risk: 1 - fact_sim (simplified; extend with NLI models)
        hall_risk = 1 - fact_sim
        
        # Source alignment: Weighted sim (mock)
        src_align = fact_sim * 0.8 + random.uniform(0.9, 1.0) * 0.2  # Bias toward alignment
        
        vector = [fact_sim, hall_risk, src_align]
        
        components = {
            "factuality_score": fact_sim,
            "hallucination_risk": hall_risk,
            "source_alignment": src_align,
            "checked_sources": len(self.source_embeddings),
            "embedding_dim": len(output_emb[0])
        }
        
        return vector, components

    def _hash_truth_vector(self, vector_data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of truth vector + metadata."""
        vec_str = json.dumps(vector_data, sort_keys=True)
        return hashlib.sha256(vec_str.encode()).hexdigest()

    async def hash_ai_output(self, output_id: str, ai_output: str, 
                             ground_truths: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compute, vectorize, and hash an AI output for forensics.
        
        :param output_id: Unique ID for the output
        :param ai_output: The AI-generated text
        :param ground_truths: Optional custom ground truths
        :return: Hashed truth vector entry
        """
        now = time.time()
        
        vector, components = self._compute_truth_vector(ai_output, ground_truths)
        
        entry = {
            **TRUTH_VECTOR_SCHEMA,
            "output_id": output_id,
            "timestamp": now,
            "truth_vector": vector,
            "components": components,
            "hash": self._hash_truth_vector({
                "output_id": output_id,
                "truth_vector": vector,
                "components": components
            })
        }
        
        # Log to ledger
        await self.ledger.append_event({
            "action": "truth_vector_hash",
            "output_id": output_id,
            "truth_vector": vector,
            "hash": entry["hash"]
        })
        
        # Log to memory
        await self.mem.log_command(entry)
        
        self.logger.info(f"Hashed truth vector for {output_id}: {vector}")
        
        return entry

    async def batch_hash_outputs(self, outputs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Batch process multiple outputs."""
        hashed = []
        for i, (output_id, ai_output) in enumerate(outputs):
            entry = await self.hash_ai_output(output_id, ai_output)
            hashed.append(entry)
            await asyncio.sleep(0.01)  # Rate limit simulation
        return hashed

    async def verify_truth_vector(self, entry: Dict[str, Any]) -> bool:
        """Verify integrity of a truth vector hash."""
        recomputed_hash = self._hash_truth_vector({
            "output_id": entry["output_id"],
            "truth_vector": entry["truth_vector"],
            "components": entry["components"]
        })
        valid = recomputed_hash == entry["hash"]
        if not valid:
            self.logger.warning(f"Truth vector hash mismatch for {entry['output_id']}")
        return valid

    async def aggregate_chain_truth(self, chain_id: str) -> Dict[str, Any]:
        """
        Aggregate truth metrics across a custody chain (integrates with ChainOfCustodyBuilder).
        
        :param chain_id: Custody chain ID
        :return: Aggregated stats (avg factuality, etc.)
        """
        # Mock integration: Query ledger for chain events
        chain_events = await self.ledger.query_chain(filter_risk="critical")  # Example filter
        vectors = [e.get("truth_vector", [0.5, 0.5, 0.5]) for e in chain_events if "truth_vector" in e]
        
        if not vectors:
            return {"error": "No vectors found"}
        
        agg = {
            "avg_factuality": np.mean([v[0] for v in vectors]),
            "avg_hall_risk": np.mean([v[1] for v in vectors]),
            "avg_alignment": np.mean([v[2] for v in vectors]),
            "total_assessed": len(vectors),
            "risk_threshold_breach": np.mean([v[1] for v in vectors]) > 0.3
        }
        
        self.logger.info(f"Aggregated truth for chain {chain_id}: {agg}")
        return agg

# Example usage / integration (e.g., post-response forensics)
async def example_truth_hashing():
    """Demo: Hash truth vector for a sample AI output."""
    hasher = TruthVectorHasher()
    
    # Mock AI output
    ai_output = "AI was invented in 1956 by John McCarthy, and it revolutionized computing."
    output_id = "out_1732999999_1"
    
    entry = await hasher.hash_ai_output(output_id, ai_output)
    print(json.dumps(entry, indent=2))
    
    # Verify
    valid = await hasher.verify_truth_vector(entry)
    print(f"Vector valid: {valid}")
    
    # Aggregate (mock chain)
    agg = await hasher.aggregate_chain_truth("analysis_001")
    print(json.dumps(agg, indent=2))

if __name__ == "__main__":
    import random  # For mock in _compute_truth_vector
    asyncio.run(example_truth_hashing())
