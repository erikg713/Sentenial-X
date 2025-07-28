# src/sentenial_x/ai/utils.py
import json

def serialize_session(session_obj) -> str:
    """Convert session model to compact JSON for the LLM prompt."""
    return json.dumps({
        "src_ip": session_obj.src_ip,
        "dst_ip": session_obj.dst_ip,
        "method": session_obj.method,
        "uri": session_obj.uri,
        "headers": session_obj.headers,
        "payload": session_obj.payload_snippet,
        "timestamp": session_obj.timestamp.isoformat()
    }, ensure_ascii=False)

