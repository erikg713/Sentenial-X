import re

signatures = {
    "sql injection": [
        r"(?i)(\bUNION\b|\bSELECT\b|--|\bor\b\s+1=1)",
    ],
    "xss": [
        r"(?i)<script.*?>.*?</script>",
        r"(?i)onerror=",
    ],
    "command injection": [
        r"(;|&&|\|\|)\s*(ls|cat|whoami|id)"
    ]
}

def detect_signatures(text: str):
    findings = []
    for category, patterns in signatures.items():
        for pattern in patterns:
            if re.search(pattern, text):
                findings.append(category)
                break
    return findings or ["none"]
