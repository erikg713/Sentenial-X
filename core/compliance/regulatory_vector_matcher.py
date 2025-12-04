# core/compliance/regulatory_vector_matcher.py
"""
Sentenial-X — Regulatory Vector Matcher

Purpose
-------
This module provides a lightweight, dependency-free engine to match code/text
(e.g. module docs, AST-extracted docstrings, semantic analyzer tags) against
a set of regulatory requirements (a "regulatory vector set").

Design goals
------------
- No external dependencies (pure Python).
- Reasonable out-of-the-box text-matching using TF-IDF + cosine similarity.
- Pluggable: accept external embedding callback (for higher-quality embeddings).
- Explainable: returns matches with rule ids, scores, and a short rationale.
- Robust: configuration, thresholds, and basic caching.

Typical usage
-------------
- Load regulation rules from a dict/list (or JSON file).
- Index the rule corpus.
- Match arbitrary text (source code comments, auditor notes, AST summaries).
- Get ranked matches and simple explanations.

Data model (Regulation JSON example)
-----------------------------------
[
    {
        "id": "gdpr:art-6",
        "title": "Lawful basis for processing",
        "text": "Processing shall be lawful only if and to the extent that at least one of the following applies...",
        "tags": ["gdpr", "consent", "legal-basis"]
    },
    ...
]

Notes
-----
- This is a best-effort matcher intended for tooling and triage. For production,
  you should replace the default tokenizer/vectorizer with high-quality
  embeddings (use `embedding_fn`) and tune thresholds.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from math import sqrt
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Any
import json
import re
import logging
from pathlib import Path
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ----------------------
# Data classes
# ----------------------
@dataclass
class RegulationRule:
    id: str
    title: str
    text: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Match:
    rule_id: str
    score: float  # cosine similarity in [0,1]
    snippet: str  # short excerpt from the rule that contributed most (best matching n-gram)
    rationale: str  # human readable explanation


# ----------------------
# Simple tokenizer / cleaning utils
# ----------------------
_WORD_RE = re.compile(r"[A-Za-z0-9_]+", flags=re.UNICODE)


def _normalize_text(text: str) -> str:
    """Lowercase and normalize whitespace."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _tokenize(text: str) -> List[str]:
    """Simple alphanumeric tokenizer returning lowercase tokens."""
    if not text:
        return []
    text = _normalize_text(text)
    return _WORD_RE.findall(text)


# ----------------------
# TF-IDF vectorizer (simple)
# ----------------------
class SimpleTfidfVectorizer:
    """
    Very small TF-IDF implementation.

    Methods:
      - fit(corpus) -> builds vocabulary and idf
      - transform(docs) -> list of sparse vectors (dict index->value)
      - fit_transform(corpus) -> convenience

    Vocabulary maps token -> index.
    """

    def __init__(self, min_df: int = 1, max_features: Optional[int] = None) -> None:
        self.min_df = max(1, int(min_df))
        self.max_features = int(max_features) if max_features is not None else None
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[int, float] = {}
        self._fitted = False

    def fit(self, docs: Iterable[str]) -> "SimpleTfidfVectorizer":
        token_docs = [list(dict.fromkeys(_tokenize(d))) for d in docs]  # unique tokens per doc for df
        df: Dict[str, int] = {}
        for tokens in token_docs:
            for t in tokens:
                df[t] = df.get(t, 0) + 1

        # filter by min_df
        items = [(t, cnt) for t, cnt in df.items() if cnt >= self.min_df]
        # sort by df desc then token
        items.sort(key=lambda x: (-x[1], x[0]))
        if self.max_features:
            items = items[: self.max_features]

        self.vocab = {t: idx for idx, (t, _) in enumerate(items)}
        N = len(token_docs) if token_docs else 1
        self.idf = {}
        for t, idx in self.vocab.items():
            df_t = df.get(t, 0)
            # smooth idf
            self.idf[idx] = float(1.0 + (N / (1.0 + df_t)))
        self._fitted = True
        logger.debug("Vectorizer fitted: vocab_size=%d", len(self.vocab))
        return self

    def _tf(self, tokens: List[str]) -> Dict[int, float]:
        counts: Dict[int, int] = {}
        for t in tokens:
            if t in self.vocab:
                counts[self.vocab[t]] = counts.get(self.vocab[t], 0) + 1
        if not counts:
            return {}
        # compute tf (term-frequency normalized)
        max_count = max(counts.values())
        return {idx: cnt / max_count for idx, cnt in counts.items()}

    def transform(self, docs: Iterable[str]) -> List[Dict[int, float]]:
        if not self._fitted:
            raise RuntimeError("Vectorizer not fitted. Call fit(...) first.")
        out: List[Dict[int, float]] = []
        for d in docs:
            tokens = _tokenize(d)
            tf = self._tf(tokens)
            tfidf = {idx: tf_val * self.idf.get(idx, 0.0) for idx, tf_val in tf.items()}
            out.append(tfidf)
        return out

    def fit_transform(self, docs: Iterable[str]) -> List[Dict[int, float]]:
        self.fit(docs)
        return self.transform(docs)


# ----------------------
# Utilities
# ----------------------
def _cosine_sim_sparse(a: Dict[int, float], b: Dict[int, float]) -> float:
    if not a or not b:
        return 0.0
    # dot product over intersection
    if len(a) < len(b):
        small, large = a, b
    else:
        small, large = b, a
    dot = 0.0
    for k, v in small.items():
        if k in large:
            dot += v * large[k]
    norm_a = sqrt(sum(v * v for v in a.values()))
    norm_b = sqrt(sum(v * v for v in b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


def _top_tokens_contributing(a: Dict[int, float], b: Dict[int, float], vocab_inv: Dict[int, str], top_n: int = 3) -> List[str]:
    # compute token contributions (abs product)
    contributions: List[Tuple[str, float]] = []
    for idx, v in a.items():
        if idx in b:
            contributions.append((vocab_inv.get(idx, "<?>"), abs(v * b[idx])))
    contributions.sort(key=lambda x: -x[1])
    return [t for t, _ in contributions[:top_n]]


# ----------------------
# Matcher
# ----------------------
class RegulatoryVectorMatcher:
    """
    Main class to register regulation rules and perform matching.

    Parameters
    ----------
    rules: Optional[Sequence[Mapping]]: initial rules to load (dicts or RegulationRule)
    min_df: int: minimum doc-frequency for a term to enter the vocabulary
    max_features: optional maximum vocabulary size
    embedding_fn: optional callable(texts: Sequence[str]) -> Sequence[Sequence[float]]
                  if provided, this will be used instead of the internal TF-IDF engine.
                  The embedding function must return a dense vector for each input text.
    """

    def __init__(
        self,
        rules: Optional[Sequence[Any]] = None,
        min_df: int = 1,
        max_features: Optional[int] = 20000,
        embedding_fn: Optional[Callable[[Sequence[str]], Sequence[Sequence[float]]]] = None,
    ) -> None:
        self._rules: List[RegulationRule] = []
        self._id_to_index: Dict[str, int] = {}
        self._vectorizer = SimpleTfidfVectorizer(min_df=min_df, max_features=max_features)
        self._doc_vectors: List[Dict[int, float]] = []
        self._embedding_fn = embedding_fn
        self._dense_vectors_cache: Optional[List[List[float]]] = None  # used when embedding_fn present
        if rules:
            self.load_rules(rules)

    # ------------------
    # Loading / indexing
    # ------------------
    def load_rules(self, rules: Sequence[Any]) -> None:
        """
        Load a sequence of rules. Each rule may be:
          - RegulationRule instance
          - dict with keys 'id', 'title', 'text', optional 'tags' and 'metadata'
        This rebuilds the index/vector space.
        """
        self._rules = []
        for r in rules:
            if isinstance(r, RegulationRule):
                rr = r
            elif isinstance(r, dict):
                rr = RegulationRule(
                    id=str(r["id"]),
                    title=str(r.get("title", "")),
                    text=str(r.get("text", "")),
                    tags=list(r.get("tags", [])),
                    metadata=dict(r.get("metadata", {})),
                )
            else:
                raise TypeError("Rule must be RegulationRule or dict")
            if rr.id in self._id_to_index:
                # overwrite existing
                logger.debug("Overwriting existing rule id=%s", rr.id)
            self._id_to_index[rr.id] = len(self._rules)
            self._rules.append(rr)

        # Rebuild vector space
        docs = [f"{r.title}\n{r.text}" for r in self._rules]
        if self._embedding_fn:
            dense = list(self._embedding_fn(docs))
            # normalize dense vectors to unit length
            self._dense_vectors_cache = [_normalize_dense_vector(v) for v in dense]
            self._doc_vectors = []  # clear sparse vectors
            logger.info("Loaded %d rules using external embedding_fn", len(self._rules))
        else:
            self._vectorizer.fit(docs)
            self._doc_vectors = self._vectorizer.transform(docs)
            self._dense_vectors_cache = None
            logger.info("Loaded %d rules into TF-IDF index (vocab=%d)", len(self._rules), len(self._vectorizer.vocab))

    def load_rules_from_json(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Rules file not found: {path}")
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("Rules JSON must be a list of rule objects")
        self.load_rules(data)

    # ------------------
    # Matching helpers
    # ------------------
    def _vectorize_query(self, text: str) -> Tuple[Optional[Dict[int, float]], Optional[List[float]]]:
        """
        Return (sparse_vector, dense_vector). Only one will be non-None depending on whether
        embedding_fn is configured.
        """
        if self._embedding_fn:
            dense = list(self._embedding_fn([text])[0])
            return None, _normalize_dense_vector(dense)
        sparse = self._vectorizer.transform([text])[0]
        return sparse, None

    def match(
        self,
        text: str,
        top_k: int = 5,
        min_score: float = 0.15,
        include_rationale: bool = True,
    ) -> List[Match]:
        """
        Match input text against the indexed rules.

        Returns up to top_k matches with score >= min_score, sorted descending by score.
        """
        sparse_q, dense_q = self._vectorize_query(text)
        results: List[Tuple[int, float]] = []

        if dense_q is not None and self._dense_vectors_cache is not None:
            # dense cosine sim
            for idx, rv in enumerate(self._dense_vectors_cache):
                score = _cosine_sim_dense(dense_q, rv)
                results.append((idx, score))
        elif sparse_q is not None:
            for idx, rv in enumerate(self._doc_vectors):
                score = _cosine_sim_sparse(sparse_q, rv)
                results.append((idx, score))
        else:
            raise RuntimeError("Matcher not properly indexed")

        # filter and sort
        results = [(i, s) for (i, s) in results if s >= min_score]
        results.sort(key=lambda x: -x[1])
        out: List[Match] = []
        vocab_inv = {idx: tok for tok, idx in self._vectorizer.vocab.items()} if self._vectorizer else {}
        for idx, score in results[:top_k]:
            rule = self._rules[idx]
            # build snippet: pick sentences from rule.text containing top tokens (best-effort)
            snippet = _extract_best_snippet(rule.text, sparse_q, vocab_inv) if include_rationale and sparse_q is not None else (rule.text[:160] + "..." if len(rule.text) > 160 else rule.text)
            rationale = f"Matched rule '{rule.title}' (id={rule.id}) with score={score:.3f}."
            if include_rationale:
                rationale += f" Tags: {', '.join(rule.tags) or 'none'}."
            out.append(Match(rule_id=rule.id, score=round(float(score), 4), snippet=snippet, rationale=rationale))
        return out

    def explain_match(self, text: str, rule_id: str) -> Optional[Dict[str, Any]]:
        """
        Produce an explanation of how the given text matched the rule (term overlaps).
        Returns None if rule_id unknown.
        """
        if rule_id not in self._id_to_index:
            return None
        idx = self._id_to_index[rule_id]
        rule = self._rules[idx]
        sparse_q, dense_q = self._vectorize_query(text)
        if sparse_q is None and dense_q is not None:
            # limited explanation for dense vectors
            return {"rule_id": rule_id, "title": rule.title, "explanation": "Matched via external dense embeddings; token-level explanation not available."}
        if sparse_q is None:
            return None
        rule_vec = self._doc_vectors[idx]
        vocab_inv = {i: t for t, i in self._vectorizer.vocab.items()}
        tokens = _top_tokens_contributing(sparse_q, rule_vec, vocab_inv, top_n=5)
        return {
            "rule_id": rule_id,
            "title": rule.title,
            "matched_tokens": tokens,
            "score": _cosine_sim_sparse(sparse_q, rule_vec),
            "rule_snippet": rule.text[:300] + ("..." if len(rule.text) > 300 else ""),
        }


# ----------------------
# Dense helpers (for embedding_fn)
# ----------------------
def _normalize_dense_vector(v: Sequence[float]) -> List[float]:
    if not v:
        return []
    norm = sqrt(sum(x * x for x in v))
    if norm == 0.0:
        return [0.0 for _ in v]
    return [float(x / norm) for x in v]


def _cosine_sim_dense(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        # fallback: try to compare on min length (rare). Better to use consistent embeddings.
        L = min(len(a), len(b))
        dot = sum(a[i] * b[i] for i in range(L))
        norm_a = sqrt(sum(a[i] * a[i] for i in range(L)))
        norm_b = sqrt(sum(b[i] * b[i] for i in range(L)))
    else:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sqrt(sum(x * x for x in a))
        norm_b = sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


def _extract_best_snippet(rule_text: str, sparse_q: Optional[Dict[int, float]], vocab_inv: Dict[int, str], window_chars: int = 160) -> str:
    """
    Heuristic: find the sentence or substring in the rule_text that contains the largest number
    of tokens from the query (using vocab_inv mapping). Fallback to start of text.
    """
    if not sparse_q:
        return rule_text[:window_chars] + ("..." if len(rule_text) > window_chars else "")
    query_tokens = set(vocab_inv[idx] for idx in sparse_q.keys() if idx in vocab_inv)
    # naive sentence split
    sentences = re.split(r'(?<=[\.\?\!;\n])\s+', rule_text)
    best_sent = ""
    best_count = -1
    for s in sentences:
        s_tokens = set(_tokenize(s))
        count = sum(1 for t in s_tokens if t in query_tokens)
        if count > best_count:
            best_count = count
            best_sent = s
    if not best_sent:
        best_sent = rule_text
    # ensure length
    if len(best_sent) > window_chars:
        best_sent = best_sent[:window_chars].rsplit(" ", 1)[0] + "..."
    return best_sent.strip()


# ----------------------
# Simple CLI for quick testing
# ----------------------
def _demo_rules() -> List[Dict[str, Any]]:
    return [
        {
            "id": "gdpr:art-6",
            "title": "Lawful basis for processing",
            "text": "Personal data shall be processed lawfully, fairly and in a transparent manner in relation to the data subject. Processing shall be lawful only if at least one of the following applies: consent, contract, legal obligation, vital interests, public task or legitimate interests.",
            "tags": ["gdpr", "privacy", "legal-basis"],
        },
        {
            "id": "hipaa:security-rule",
            "title": "HIPAA Security Rule",
            "text": "Covered entities must maintain administrative, physical, and technical safeguards to protect electronic protected health information (ePHI). Safeguards include access controls, audit controls, integrity controls, and transmission security.",
            "tags": ["hipaa", "security", "ephi"],
        },
        {
            "id": "pci:dss-3.2",
            "title": "PCI DSS - Protect Cardholder Data",
            "text": "Cardholder data must be protected; storage of cardholder data must be minimized and sensitive authentication data must not be stored after authorization. Use encryption and strong access controls.",
            "tags": ["pci", "payment", "cardholder"],
        },
    ]


def _demo_cli_main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="regulatory-vector-matcher")
    parser.add_argument("file", nargs="?", help="Optional file containing text to match (defaults to stdin)")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=0.12)
    args = parser.parse_args(argv)

    if args.file:
        p = Path(args.file)
        if not p.exists():
            print(f"File not found: {args.file}")
            return 2
        text = p.read_text(encoding="utf-8")
    else:
        print("Enter text to match (finish with EOF / Ctrl-D):")
        text = ""
        try:
            import sys
            text = sys.stdin.read()
        except KeyboardInterrupt:
            return 0

    matcher = RegulatoryVectorMatcher(rules=_demo_rules(), min_df=1, max_features=5000)
    matches = matcher.match(text, top_k=args.top_k, min_score=args.min_score, include_rationale=True)

    if not matches:
        print("No matches found (below threshold).")
        return 0

    print("Matches:")
    for m in matches:
        print(f"- rule_id={m.rule_id} score={m.score:.3f}")
        print(f"  snippet: {m.snippet}")
        print(f"  rationale: {m.rationale}")
        print()
    return 0


# ----------------------
# Unit-style quick tests (run when module executed)
# ----------------------
if __name__ == "__main__":
    # Quick smoke test
    test_text = (
        "We store card numbers and authentication data after authorization. We must ensure "
        "encryption and strong access control over stored cardholder data."
    )
    matcher = RegulatoryVectorMatcher(rules=_demo_rules())
    print("Matching test_text ->")
    for m in matcher.match(test_text, top_k=5, min_score=0.05):
        print(f"{m.rule_id:20} score={m.score:.3f} snippet={m.snippet!r}")
    # If run as CLI
    import sys
    if len(sys.argv) > 1:
        sys.exit(_demo_cli_main(sys.argv[1:]))
