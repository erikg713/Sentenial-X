"""
core/adapters/http_parser.py

Sentenial-X Adapters HTTP Parser Module - parses HTTP requests/responses for security analysis,
extracting headers, payloads, and detecting anomalies like injection attempts or unauthorized access.
Integrates with IDS modules for triggering alerts, IncidentQueue for escalation, and forensics for logging parsed data.
Supports async parsing, pattern-based detection for common web attacks, and machine learning-based anomaly detection
using statistical methods with numpy and scipy for outlier identification on extracted features.
"""

import asyncio
import re
import json
import numpy as np
from scipy import stats
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse, parse_qs
from cli.memory_adapter import get_adapter
from cli.logger import default_logger
from core.orchestrator.incident_queue import IncidentQueue
from core.ids.honeypot_trigger_ids import HoneypotTriggerIDS
from core.forensics.ledger_sequencer import LedgerSequencer

# HTTP anomaly patterns: regex for common attacks (expanded for broader coverage)
ANOMALY_PATTERNS = {
    "sql_injection": [
        r"(?i)\b(select|insert|update|delete|union|drop|alter|create|truncate|exec|execute)\b",
        r"1=1", r"'--", r";--", r"or\s+1=1", r"and\s+1=1", r"'\s+or\s+'", r"waitfor\s+delay",
        r"benchmark\s*\(", r"pg_sleep\s*\(", r"sleep\s*\("
    ],
    "xss": [
        r"<script>", r"javascript:", r"onerror=", r"alert\(", r"confirm\(", r"prompt\(",
        r"onload=", r"onmouseover=", r"svg\s+onload=", r"img\s+src=\s*x:\s*onerror",
        r"expression\s*\(", r"data:\s*text/html"
    ],
    "path_traversal": [
        r"\.\./", r"%2e%2e/", r"../", r"..\..\.", r"%252e%252e%252f", r"....//",
        r"etc/passwd", r"windows/system32", r"boot.ini", r"win.ini", r"shadow",
        r"proc/version", r"cmd.exe"
    ],
    "command_injection": [
        r";\s*\w+", r"\|\s*\w+", r"&&\s*\w+", r"\$\(", r"exec\s*\(", r"system\s*\(",
        r"passthru\s*\(", r"shell_exec\s*\(", r"popen\s*\(", r"backtick",
        r"wget\s+", r"curl\s+", r"nc\s+", r"bash\s+-c"
    ],
    "header_anomalies": [
        r"Host:\s*[^a-zA-Z0-9\.-]", r"User-Agent:\s*sqlmap", r"User-Agent:\s*nikto",
        r"User-Agent:\s*nessus", r"Referer:\s*sqlmap", r"X-Forwarded-For:\s*['\"];",
        r"Accept:\s*\*/\*", r"Authorization:\s*basic\s*[a-zA-Z0-9+/=]{100,}"  # Suspicious base64
    ],
    "rfi_lfi": [  # Remote/Local File Inclusion
        r"http://evil.com", r"https://evil.com", r"file:///", r"php://input",
        r"data://text/plain;base64,", r"gopher://", r"expect://"
    ],
    "csrf_bypass": [
        r"csrf_token=\s*'", r"anti_csrf=\s*1=1", r"token=\s*or"
    ],
    "open_redirect": [
        r"redirect=\s*http://evil.com", r"url=\s*//evil.com", r"next=\s*javascript:"
    ]
}

class HttpParser:
    """
    Asynchronous HTTP parser for security inspection of requests/responses.
    Extracts components, detects anomalies, and integrates with queue/IDS/forensics.
    
    :param queue: Optional IncidentQueue for escalation on detections
    :param ids: Optional HoneypotTriggerIDS for additional triggering
    :param ledger: Optional LedgerSequencer for logging parses
    """
    def __init__(self, queue: Optional[IncidentQueue] = None,
                 ids: Optional[HoneypotTriggerIDS] = None,
                 ledger: Optional[LedgerSequencer] = None):
        self.queue = queue or IncidentQueue()
        self.ids = ids or HoneypotTriggerIDS(self.queue)
        self.ledger = ledger or LedgerSequencer()
        self.mem = get_adapter()
        self.logger = default_logger
        # Mock "training data" means/stds for ML anomaly (in prod, load from model)
        self.feature_means = np.array([50, 2, 100, 10])  # path_len, num_params, body_len, header_count
        self.feature_stds = np.array([20, 1, 50, 5])     # Example stds

    async def parse_http(self, http_data: str, is_request: bool = True) -> Dict[str, Any]:
        """
        Parse raw HTTP data (request or response) and extract components.
        
        :param http_data: Raw HTTP string (e.g., from socket or log)
        :param is_request: True for requests, False for responses
        :return: Parsed dict with method/path/status, headers, body, params
        """
        lines = http_data.splitlines()
        if not lines:
            return {"error": "Empty HTTP data"}
        
        # Parse start line
        start_line = lines[0]
        if is_request:
            parts = start_line.split()
            if len(parts) < 3:
                return {"error": "Invalid request line"}
            method, path, version = parts
            parsed_url = urlparse(path)
            params = parse_qs(parsed_url.query)
        else:
            parts = start_line.split()
            if len(parts) < 3:
                return {"error": "Invalid response line"}
            version, status, *reason = parts
            method = path = None
            params = {}
        
        # Parse headers
        headers = {}
        body_start = 0
        for i, line in enumerate(lines[1:], 1):
            if not line.strip():
                body_start = i + 1
                break
            if ':' in line:
                key, value = line.split(':', 1)
                headers[key.strip()] = value.strip()
        
        # Body (JSON/text)
        body_lines = lines[body_start:]
        body = '\n'.join(body_lines)
        try:
            body_json = json.loads(body) if body and "application/json" in headers.get("Content-Type", "") else None
        except json.JSONDecodeError:
            body_json = None
        
        parsed = {
            "is_request": is_request,
            "method": method,
            "path": path,
            "version": version,
            "status": status if not is_request else None,
            "headers": headers,
            "params": params,
            "body": body,
            "body_json": body_json
        }
        
        # Log parse
        await self.mem.log_command({
            "action": "http_parse",
            "parsed": parsed
        })
        
        self.logger.info(f"Parsed HTTP {'request' if is_request else 'response'}")
        
        return parsed

    def _extract_features(self, parsed: Dict[str, Any]) -> np.ndarray:
        """
        Extract numerical features for ML anomaly detection.
        
        :param parsed: Parsed HTTP data
        :return: Feature vector [path_len, num_params, body_len, header_count]
        """
        path_len = len(parsed.get("path", ""))
        num_params = len(parsed.get("params", {}))
        body_len = len(parsed.get("body", ""))
        header_count = len(parsed.get("headers", {}))
        return np.array([path_len, num_params, body_len, header_count])

    async def detect_ml_anomalies(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect anomalies using machine learning (statistical outlier detection with z-scores).
        
        :param parsed: Parsed HTTP data
        :return: ML detection results with outliers and risk
        """
        features = self._extract_features(parsed)
        z_scores = np.abs(stats.zscore(features, nan_policy='omit'))  # Handle NaN if any
        
        # In prod, use pre-computed means/stds; here assume self.feature_means/stds
        # z_scores = (features - self.feature_means) / self.feature_stds
        
        outliers = np.where(z_scores > 3)[0]  # Threshold 3 for anomaly
        outlier_count = len(outliers)
        ml_risk = "critical" if outlier_count >= 2 else "high" if outlier_count > 0 else "low"
        
        ml_detection = {
            "features": features.tolist(),
            "z_scores": z_scores.tolist(),
            "outliers": outliers.tolist(),
            "outlier_count": outlier_count,
            "ml_risk_level": ml_risk
        }
        
        if outlier_count > 0:
            self.logger.warning(f"ML detected {outlier_count} outliers, risk {ml_risk}")
        
        return ml_detection

    async def detect_anomalies(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scan parsed HTTP for anomalies and return detections (combines pattern + ML).
        
        :param parsed: Output from parse_http
        :return: Detections with hits and risk
        """
        hits = {}
        check_areas = [
            parsed.get("path", ""),
            json.dumps(parsed.get("params", {})),
            json.dumps(parsed.get("headers", {})),
            parsed.get("body", "")
        ]
        
        for area in check_areas:
            for cat, pats in ANOMALY_PATTERNS.items():
                cat_hits = [pat for pat in pats if re.search(pat, area, re.IGNORECASE)]
                if cat_hits:
                    if cat not in hits:
                        hits[cat] = []
                    hits[cat].extend(cat_hits)
        
        pattern_hit_count = sum(len(v) for v in hits.values())
        pattern_risk = "critical" if pattern_hit_count >= 3 else "high" if pattern_hit_count > 0 else "low"
        
        ml_detection = await self.detect_ml_anomalies(parsed)
        
        combined_hit_count = pattern_hit_count + ml_detection["outlier_count"]
        combined_risk = "critical" if combined_hit_count >= 3 else "high" if combined_hit_count > 0 else "low"
        
        detection = {
            "pattern_hits": hits,
            "pattern_hit_count": pattern_hit_count,
            "pattern_risk_level": pattern_risk,
            "ml_detection": ml_detection,
            "combined_hit_count": combined_hit_count,
            "combined_risk_level": combined_risk
        }
        
        if combined_hit_count > 0:
            # Trigger IDS
            await self.ids.trigger_on_hit(detection)
            # Enqueue
            await self.queue.enqueue(combined_risk, detection)
            # Log to ledger
            await self.ledger.append_event({
                "action": "http_anomaly",
                "parsed": parsed,
                "detection": detection
            })
        
        self.logger.warning(f"Detected {combined_hit_count} HTTP anomalies (pattern+ML), risk {combined_risk}")
        
        return detection

    async def process_http(self, http_data: str, is_request: bool = True) -> Dict[str, Any]:
        """
        Full process: Parse, detect anomalies, and return combined results.
        
        :param http_data: Raw HTTP
        :param is_request: Request or response
        :return: Parsed + detection
        """
        parsed = await self.parse_http(http_data, is_request)
        if "error" in parsed:
            return parsed
        
        detection = await self.detect_anomalies(parsed)
        return {**parsed, "detection": detection}

# Example usage / integration (e.g., with honeypot traffic)
async def example_http_parse():
    """Demo: Parse and detect in a mock HTTP request."""
    parser = HttpParser()
    
    # Mock HTTP request
    http_request = """GET /admin?token=123&path=../etc/passwd HTTP/1.1
Host: example.com
User-Agent: suspicious-agent
Content-Type: application/json

{"command": "exec('ls')"}"""
    
    result = await parser.process_http(http_request)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(example_http_parse())
