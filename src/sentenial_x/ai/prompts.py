# src/sentenial_x/ai/prompts.py

HTTP_ANALYSIS = """
You are a cyber-security analyst. Analyze the following HTTP session metadata and
payload. Identify anomalies, map them to ATT&CK techniques, and suggest a risk score.
Session details:
{session_json}

Your response must be valid JSON with keys:
- findings: list of {{ technique_id, description, severity }}
- risk_score: integer between 0–100
- remediation: free-text suggestions
"""

CLASSIFY_UA = """
You are an agent that classifies User-Agent strings. Given UA:
{user_agent}
Return JSON: {{category: one of [‘browser’, ‘scanner’, ‘bot’, ‘malformed’]}}
"""

