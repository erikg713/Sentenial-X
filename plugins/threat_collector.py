from plugins.plugin_base import Plugin
from typing import Any, Dict, List
import json
import numpy as np
from sklearn.decomposition import PCA

# reuse your existing functions here (or import them)
def collect_sample(file_path: str, suspicious_strings: bool = False) -> Dict[str,Any]:
    # your real implementation goes here
    return {
        "file": file_path,
        "matched_suspicious_strings": ["powershell.exe -EncodedCommand"] if suspicious_strings else [],
        "cve_ids": ["CVE-2023-12345"]
    }

def get_embedding(text: str) -> List[float]:
    # your real implementation goes here
    np.random.seed(len(text))
    return np.random.rand(64).tolist()

class ThreatCollectorPlugin(Plugin):
    name = "threat_collector"
    description = "Collect metadata and embeddings from a single file sample"
    
    # GUI form metadata
    parameters: List[Dict[str,Any]] = [
      {
        "name": "file_path",
        "type": "file",
        "label": "Select Threat Sample",
        "dialog": {"mode": "open", "filter": "All Files (*)"}
      },
      {
        "name": "suspicious_strings",
        "type": "bool",
        "label": "Enable Suspicious‐String Detection",
        "default": True
      }
    ]

    def run(self, file_path: str, suspicious_strings: bool = True) -> Dict[str,Any]:
        # 1) collect metadata
        sample = collect_sample(file_path, suspicious_strings)
        # 2) read up to N bytes
        with open(file_path, "r", errors="ignore") as f:
            text = f.read(10000)
        # 3) compute embedding
        emb = get_embedding(text)
        # 4) return full payload
        return {
            "sample_info": sample,
            "embedding_preview": emb[:10], 
            "full_embedding": emb
        }

def register():
    return ThreatCollectorPlugin()
