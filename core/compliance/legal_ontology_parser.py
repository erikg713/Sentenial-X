# core/compliance/legal_ontology_parser.py

import re
from typing import Dict, List

# Known regulations to detect
REGULATIONS = [
    "GDPR",
    "HIPAA",
    "CCPA",
    "SOX",
    "FERPA",
    "PCI-DSS"
]

# Map high-level actions to keyword patterns
ACTION_KEYWORDS = {
    "Data Erasure": [
        r"\berasable\b",
        r"\berasure\b",
        r"\berase\b",
        r"\bdelete\b",
        r"\bdeleted\b"
    ],
    "Data Encryption": [
        r"\bencrypt\b",
        r"\bencrypted\b",
        r"\bencryption\b"
    ],
    "Access Control": [
        r"\baccess control\b",
        r"\bauthoriz(?:ation|e)\b",
        r"\bpermission\b"
    ],
    "Data Retention": [
        r"\bretain\b",
        r"\bretention\b",
        r"\bstore\b"
    ]
}


def parse_policy(text: str) -> Dict[str, any]:
    """
    Scan a free-form policy string for regulatory mentions
    and required compliance actions.

    Returns a dict with:
      - "regulations": list of detected regs (e.g., ["GDPR"])
      - "action_required": a single best-match action (e.g., "Data Erasure")
    """
    regs: List[str] = []
    for reg in REGULATIONS:
        if re.search(rf"\b{re.escape(reg)}\b", text, re.IGNORECASE):
            regs.append(reg)

    action_required = None
    for action, patterns in ACTION_KEYWORDS.items():
        for pat in patterns:
            if re.search(pat, text, re.IGNORECASE):
                action_required = action
                break
        if action_required:
            break

    return {
        "regulations": regs,
        "action_required": action_required
    }
