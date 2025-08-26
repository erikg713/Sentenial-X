"""
High-quality, robust tests for LLM accuracy and evaluation helpers.

This test module provides:
- Reference implementations for common evaluation metrics (exact match, token-F1, top-k accuracy).
- Thorough parametrized test-cases covering edge cases (whitespace, punctuation, casing, empty predictions,
  multiple references, partially overlapping tokens).
- Cross-checks that attempt to import equivalent functions from the project and compare outputs
  with the reference implementations. If the project's functions are unavailable, those tests are skipped
  rather than failing, so these tests remain useful both inside and outside of the repository.

Design goals:
- Readable, maintainable, Pythonic code that looks hand-crafted by an experienced developer.
- Deterministic behavior and clear assertion messages for easy debugging.
- Minimal external dependencies (only pytest is required).
"""

from __future__ import annotations

import importlib
import math
import re
from collections import Counter
from typing import Iterable, List, Sequence, Tuple, Union, Optional

import pytest


# --------------------------
# Reference implementation
# --------------------------

_WORD_RE = re.compile(r"\w+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def _normalize_text(text: Optional[object]) -> str:
    """
    Normalize text for comparison:
    - convert to string (handles None)
    - lowercase
    - strip surrounding whitespace
    - collapse internal whitespace
    - remove simple punctuation
    This is intentionally conservative and easy to reason about in tests.
    """
    if text is None:
        return ""
    txt = str(text).lower().strip()
    # Remove punctuation (keep word characters and whitespace)
    txt = _PUNCT_RE.sub("", txt)
    # Collapse whitespace to single space
    txt = re.sub(r"\s+", " ", txt)
    return txt


def exact_match_score(prediction: Optional[object], references: Union[Sequence[object], object]) -> int:
    """
    Exact match: return 1 if normalized prediction exactly equals any normalized reference, else 0.

    Returns:
        int: 1 for exact match, 0 otherwise. Using int keeps comparisons explicit in tests.
    """
    # Allow passing a single reference string or sequence of references
    if isinstance(references, (str, bytes)) or not isinstance(references, Iterable):
        refs = [references]
    else:
        refs = list(references)

    pred_norm = _normalize_text(prediction)
    for r in refs:
        if pred_norm == _normalize_text(r):
            return 1
    return 0


def token_f1(prediction: Optional[object], references: Union[Sequence[object], object]) -> float:
    """
    Token-level F1: compute overlapping token counts against each reference and return the maximum F1.

    Behavior:
    - For multiple references, returns the maximum F1 across references.
    - Empty prediction and empty reference => F1 == 1.0
    - Empty prediction or empty reference (but not both) => F1 == 0.0

    Returns:
        float: F1 score in [0.0, 1.0]
    """
    if isinstance(references, (str, bytes)) or not isinstance(references, Iterable):
        refs = [references]
    else:
        refs = list(references)

    pred_tokens = _WORD_RE.findall(_normalize_text(prediction))
    best_f1 = 0.0

    for ref in refs:
        ref_tokens = _WORD_RE.findall(_normalize_text(ref))

        # Both empty -> perfect match
        if not pred_tokens and not ref_tokens:
            best_f1 = max(best_f1, 1.0)
            continue

        # One empty, the other not -> zero
        if not pred_tokens or not ref_tokens:
            best_f1 = max(best_f1, 0.0)
            continue

        pred_counts = Counter(pred_tokens)
        ref_counts = Counter(ref_tokens)

        # Multiset intersection size
        overlap = sum((pred_counts & ref_counts).values())

        if overlap == 0:
            f1 = 0.0
        else:
            precision = overlap / sum(pred_counts.values())
            recall = overlap / sum(ref_counts.values())
            f1 = 2 *
