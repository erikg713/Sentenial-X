# sentenialx/core/semantic_engine/parser.py

import json
from urllib.parse import urlparse, parse_qs

class SemanticParser:
    def parse_request(self, method: str, url: str, headers: dict, body: str = "") -> dict:
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        body_tokens = self._tokenize_body(headers, body)

        return {
            "method": method,
            "path": parsed_url.path,
            "query_params": query_params,
            "headers": headers,
            "body_tokens": body_tokens,
        }

    def _tokenize_body(self, headers, body):
        if "application/json" in headers.get("Content-Type", ""):
            try:
                json_body = json.loads(body)
                return self._flatten_json(json_body)
            except Exception:
                return ["malformed_json"]
        return body.split() if body else []

    def _flatten_json(self, obj, parent_key='', sep='.'):
        items = []
        for k, v in obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_json(v, new_key, sep=sep))
            else:
                items.append(f"{new_key}:{v}")
        return items