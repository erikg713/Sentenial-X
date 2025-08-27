# ===== File: core/detection/config.py =====
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class DetectionConfig:
    low_threshold: float = 0.20
    medium_threshold: float = 0.45
    high_threshold: float = 0.70
    critical_threshold: float = 0.90
    dedup_window_sec: int = 180
    max_queue_size: int = 10_000
    yara_paths: Optional[List[str]] = None
    hash_denylist_paths: Optional[List[str]] = None
    rule_paths: Optional[List[str]] = None
    ml_model_path: Optional[str] = None
    jsonl_path: Optional[str] = None
    sqlite_path: Optional[str] = None
  
