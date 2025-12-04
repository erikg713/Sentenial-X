# core/compliance/legal_ontology_parser.py
"""
Improved policy text parser for detecting regulatory mentions and
high-level compliance actions.

Goals / improvements:
- Precompile regexes for speed and clarity.
- Recognize regulation canonical names and common variants/full names.
- Return detailed matches (span, matched text, pattern) and allow
  detection of multiple actions instead of only one best match.
- Provide a simple confidence heuristic for the chosen primary action.
- More precise typing and clear, human-style code organization.
"""

from typing import Dict, List, Optional, Any, Pattern, Tuple
import re
from collections import OrderedDict

# Canonical regulation identifiers mapped to pattern variants (common abbreviations,
# full names, and other likely mentions). We keep canonical short names as keys.
REGULATION_PATTERNS = {
    "GDPR": [
        "GDPR",
        "General Data Protection Regulation",
        "data protection regulation"
    ],
    "HIPAA": [
        "HIPAA",
        "Health Insurance Portability and Accountability Act",
        "health information privacy"
    ],
    "CCPA": [
        "CCPA",
        "California Consumer Privacy Act",
        "California privacy"
    ],
    "SOX": [
        "SOX",
        "Sarbanes[- ]Oxley",
        "Sarbanes Oxley"
    ],
    "FERPA": [
        "FERPA",
        "Family Educational Rights and Privacy Act",
        "educational privacy"
    ],
    "PCI-DSS": [
        "PCI[- ]DSS",
        "Payment Card Industry Data Security Standard",
        "payment card industry"
    ]
}

# High-level actions with prioritized order (first = highest priority).
# Values are lists of regex fragments. We keep them specific and word-boundary anchored
# when appropriate. Order matters when picking a single primary action.
ACTION_KEYWORDS = OrderedDict([
    ("Data Erasure", [
        r"\berasable\b",
        r"\berasure\b",
        r"\berase\b",
        r"\bdelete\b",
        r"\bdeleted\b",
        r"\bremove personal data\b",
        r"\bright to be forgotten\b"
    ]),
    ("Data Encryption", [
        r"\bencrypt(?:ion|ed|s)?\b",
        r"\bencrypted\b",
        r"\bencryption\b",
        r"\bencrypting\b",
    ]),
    ("Access Control", [
        r"\baccess control\b",
        r"\bauthoriz(?:ation|e)\b",
        r"\bpermission(?:s)?\b",
        r"\brole[- ]based access\b",
        r"\baccess management\b"
    ]),
    ("Data Retention", [
        r"\bretain(?:ed|ing)?\b",
        r"\bretention\b",
        r"\bstore(?:d|ing)?\b",
        r"\bretention period\b",
        r"\bdata retention policy\b"
    ])
])

# Precompile compiled patterns for performance and clarity
def _compile_regulation_patterns(reg_map: Dict[str, List[str]]) -> Dict[str, Pattern]:
    compiled: Dict[str, Pattern] = {}
    for canon, variants in reg_map.items():
        # Escape tokens except those that intentionally contain regex like "Sarbanes[- ]Oxley"
        safe_variants = []
        for v in variants:
            # If variant contains character classes or quantifiers, assume it's an intentional regex
            if re.search(r"[\\\[\]\(\)\|\^\$\.\*\+\?\{\}]", v):
                safe_variants.append(v)
            else:
                safe_variants.append(re.escape(v))
        pattern = r"\b(?:" + "|".join(safe_variants) + r")\b"
        compiled[canon] = re.compile(pattern, flags=re.IGNORECASE)
    return compiled


def _compile_action_patterns(action_map: Dict[str, List[str]]) -> Dict[str, Pattern]:
    return {action: re.compile("|".join(patterns), flags=re.IGNORECASE) for action, patterns in action_map.items()}


_COMPILED_REGULATIONS = _compile_regulation_patterns(REGULATION_PATTERNS)
_COMPILED_ACTIONS = _compile_action_patterns(ACTION_KEYWORDS)


def _normalize_text(text: str) -> str:
    # Lightweight normalization: collapse whitespace, strip, keep original casing for spans.
    return re.sub(r"\s+", " ", text).strip()


def _find_matches(pattern: Pattern, text: str) -> List[Tuple[str, Tuple[int, int], str]]:
    """
    Return list of tuples: (matched_text, (start, end), pattern_source)
    pattern_source is pattern.pattern for traceability.
    """
    matches: List[Tuple[str, Tuple[int, int], str]] = []
    for m in pattern.finditer(text):
        matches.append((m.group(0), (m.start(), m.end()), pattern.pattern))
    return matches


def parse_policy(text: str, detect_multiple_actions: bool = True) -> Dict[str, Any]:
    """
    Scan a free-form policy string for regulatory mentions and required compliance actions.

    Parameters:
    - text: raw policy text to analyze.
    - detect_multiple_actions: if True, return all detected actions (with details);
      if False, return only the highest-priority/first-detected action.

    Returns a dictionary with keys:
    - regulations: List of dicts with keys {name, matched_text, span, pattern}
    - actions: List of dicts with keys {name, matched_text, span, pattern}
    - primary_action: Optional[str] chosen primary action name (None if not found)
    - confidence: float heuristic (0.0 - 1.0) expressing confidence in primary_action
    - normalized_text: normalized input (for debugging)
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "regulations": [],
            "actions": [],
            "primary_action": None,
            "confidence": 0.0,
            "normalized_text": ""
        }

    normalized = _normalize_text(text)

    regs_found: List[Dict[str, Any]] = []
    for canon, pattern in _COMPILED_REGULATIONS.items():
        for matched_text, span, pat in _find_matches(pattern, normalized):
            regs_found.append({
                "name": canon,
                "matched_text": matched_text,
                "span": span,
                "pattern": pat
            })

    # Deduplicate regulations keeping earliest appearance first
    regs_unique: Dict[str, Dict[str, Any]] = {}
    for r in sorted(regs_found, key=lambda x: x["span"][0]):
        if r["name"] not in regs_unique:
            regs_unique[r["name"]] = r
    regs_list = list(regs_unique.values())

    actions_found: List[Dict[str, Any]] = []
    # We respect the ACTION_KEYWORDS order for priority decisions.
    for action_name, compiled in _COMPILED_ACTIONS.items():
        matches = _find_matches(compiled, normalized)
        for matched_text, span, pat in matches:
            actions_found.append({
                "name": action_name,
                "matched_text": matched_text,
                "span": span,
                "pattern": pat
            })

    # Sort actions by earliest position in text; keep ordering stable for same position.
    actions_found.sort(key=lambda x: x["span"][0])

    # Build unique action list preserving first match per action
    action_by_name: Dict[str, Dict[str, Any]] = {}
    for a in actions_found:
        if a["name"] not in action_by_name:
            action_by_name[a["name"]] = a

    actions_list = list(action_by_name.values())

    primary_action: Optional[str] = None
    confidence: float = 0.0

    if actions_list:
        # If detect_multiple_actions is False, pick highest priority (order in ACTION_KEYWORDS)
        if not detect_multiple_actions:
            for name in ACTION_KEYWORDS.keys():
                if name in action_by_name:
                    primary_action = name
                    break
        else:
            # Prefer the action that appears first in the text.
            primary_action = actions_list[0]["name"]

        # Simple confidence heuristic:
        # base confidence = 0.6 if matched once, +0.15 per additional distinct action (capped).
        unique_action_count = len(actions_list)
        confidence = 0.5 + min(0.4, 0.15 * unique_action_count)
        # If a regulation is explicitly mentioned that strongly implies the action (e.g., GDPR + erasure),
        # bump confidence slightly.
        if regs_list and unique_action_count > 0:
            confidence = min(1.0, confidence + 0.05)

    # Final response
    return {
        "regulations": regs_list,
        "actions": actions_list if detect_multiple_actions else ([action_by_name[primary_action]] if primary_action else []),
        "primary_action": primary_action,
        "confidence": round(confidence, 2),
        "normalized_text": normalized
    }


# Explicit exports
__all__ = ["parse_policy", "REGULATION_PATTERNS", "ACTION_KEYWORDS"]
