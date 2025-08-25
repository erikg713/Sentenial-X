## 📘 API Reference ##

This document describes the REST APIs exposed by the Semantic Analyzer and Cortex Engine.
All APIs follow REST conventions and return JSON responses.


---

🔎 Semantic Analyzer API

POST /analyze

Performs contextual analysis on input text.

Request

{
  "text": "User attempted unauthorized access"
}

Response

{
  "entities": ["User"],
  "intent": "security_breach",
  "confidence": 0.93,
  "embedding_vector": [0.234, -0.019, ...]
}

Parameters

Field	Type	Description	Required

text	string	Raw input text	✅



---

POST /batch_analyze

Processes a batch of documents in one request.

Request

{
  "documents": [
    "Unauthorized login attempt",
    "Payment failed for transaction 123"
  ]
}

Response

[
  {
    "entities": ["login"],
    "intent": "security_breach",
    "confidence": 0.91
  },
  {
    "entities": ["transaction 123"],
    "intent": "payment_failure",
    "confidence": 0.88
  }
]


---

GET /health

Health check for service monitoring.

Response

{ "status": "ok", "uptime": 12453 }


---

🧠 Cortex Orchestration API

POST /decision

Generates decisions based on semantic input + models.

Request

{
  "input": "unauthorized access detected"
}

Response

{
  "action": "alert_admin",
  "confidence": 0.94,
  "trace_id": "abc-123"
}

Parameters

Field	Type	Description	Required

input	string	Contextual text or structured payload	✅



---

POST /batch_decision

Batch decision-making for multiple inputs.

Request

{
  "inputs": [
    "multiple failed login attempts",
    "payment declined at checkout"
  ]
}

Response

[
  {
    "action": "lock_account",
    "confidence": 0.92,
    "trace_id": "lock-456"
  },
  {
    "action": "notify_billing",
    "confidence": 0.89,
    "trace_id": "bill-789"
  }
]


---

GET /status

Returns current Cortex system status.

Response

{
  "status": "active",
  "services": {
    "semantic_analyzer": "online",
    "models": "online"
  },
  "uptime": 34823
}


---

⚙️ API Notes

All endpoints use JSON format.

Recommended Content-Type header:

Content-Type: application/json

All responses include trace IDs for observability in distributed logs.

Batch endpoints improve throughput for high-volume workloads.



---
