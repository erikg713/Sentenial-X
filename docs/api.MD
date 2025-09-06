---

### `api.md`

```markdown
# API Reference

FastAPI endpoints for programmatic access.

| Endpoint | Method | Description | Example Payload |
|----------|--------|-------------|----------------|
| `/cortex/analyze` | POST | Full AI analysis | `{"text": "suspicious log line", "features": [0.5,0.7]}` |
| `/wormgpt-detector` | POST | Adversarial AI detection | `{"text": "malicious prompt"}` |
| `/simulate` | POST | Trigger emulation | `{"text": "suspicious behavior"}` |
| `/alerts` | POST | Dispatch alerts | `{"type": "ransomware", "severity": "high"}` |
