"""
core/forensics/chain_of_custody_builder.py

Sentenial-X Forensics Chain of Custody Builder Module - constructs and verifies
immutable chains of custody for AI forensic evidence, ensuring non-repudiation,
integrity, and traceability across interactions and ledgers.
Integrates with LedgerSequencer for event linkage and digital signatures for authenticity.
"""

import asyncio
import time
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.exceptions import InvalidSignature
from cli.memory_adapter import get_adapter
from cli.logger import default_logger
from .ledger_sequencer import LedgerSequencer  # Relative import for integration

# Custody event schema
CUSTODY_SCHEMA = {
    "custody_id": str,
    "timestamp": float,
    "actor": str,  # e.g., "analyst_user_id" or "system"
    "action": str,  # e.g., "acquired", "analyzed", "transferred"
    "evidence_ref": str,  # Ledger event_id or hash
    "description": str,
    "signature": str,  # Base64-encoded digital signature
    "prev_custody_hash": str,
    "chain_hash": str  # Cumulative chain hash
}

class ChainOfCustodyBuilder:
    """
    Builds and verifies chains of custody for forensic evidence in AI interactions.
    Each custody event is signed and hashed into an immutable chain, linked to ledger entries.
    
    :param ledger: Optional LedgerSequencer instance for integration
    :param private_key_file: Path to PEM private key for signing (auto-generate if None)
    """
    def __init__(self, ledger: Optional[LedgerSequencer] = None, 
                 private_key_file: Optional[str] = None):
        self.ledger = ledger or LedgerSequencer()
        self.mem = get_adapter()
        self.logger = default_logger
        self.private_key, self.public_key = self._load_or_generate_keys(private_key_file)
        self.active_chains: Dict[str, List[Dict[str, Any]]] = {}  # chain_id -> custody events

    def _load_or_generate_keys(self, key_file: Optional[str]) -> Tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey]:
        """Load or generate RSA key pair for signing."""
        if key_file and os.path.exists(key_file):
            with open(key_file, "rb") as f:
                private_key = serialization.load_pem_private_key(f.read(), password=None)
                public_key = private_key.public_key()
        else:
            private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            public_key = private_key.public_key()
            if key_file:
                with open(key_file, "wb") as f:
                    f.write(private_key.private_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PrivateFormat.PKCS8,
                        encryption_algorithm=serialization.NoEncryption()
                    ))
                self.logger.info(f"Generated new private key: {key_file}")
        return private_key, public_key

    def _sign_event(self, event: Dict[str, Any]) -> str:
        """Sign event data with private key."""
        event_str = json.dumps(event, sort_keys=True).encode()
        signature = self.private_key.sign(
            event_str,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256()
        )
        return signature.hex()  # Hex for storage

    def _verify_signature(self, event: Dict[str, Any], signature_hex: str) -> bool:
        """Verify event signature with public key."""
        try:
            event_str = json.dumps({k: v for k, v in event.items() if k != "signature"}, sort_keys=True).encode()
            signature = bytes.fromhex(signature_hex)
            self.public_key.verify(
                signature,
                event_str,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                hashes.SHA256()
            )
            return True
        except InvalidSignature:
            return False

    def _compute_chain_hash(self, events: List[Dict[str, Any]]) -> str:
        """Compute cumulative hash for the entire chain."""
        chain_str = "".join(json.dumps(e, sort_keys=True) for e in events)
        return hashlib.sha256(chain_str.encode()).hexdigest()

    async def build_custody_event(self, actor: str, action: str, evidence_ref: str, 
                                  description: str, chain_id: str) -> Dict[str, Any]:
        """
        Build a single custody event and append to chain.
        
        :param actor: Identifier of the custodian (e.g., user ID)
        :param action: Custody action (e.g., "acquired", "reviewed")
        :param evidence_ref: Reference to ledger event or hash
        :param description: Human-readable details
        :param chain_id: Unique ID for the custody chain
        :return: Signed and hashed custody event
        """
        now = time.time()
        chain_id = chain_id or f"coc_{int(now)}"
        
        # Initialize chain if new
        if chain_id not in self.active_chains:
            self.active_chains[chain_id] = []
        
        prev_hash = self.active_chains[chain_id][-1]["chain_hash"] if self.active_chains[chain_id] else "genesis"
        
        event = {
            **CUSTODY_SCHEMA,
            "custody_id": f"coc_evt_{int(now)}_{len(self.active_chains[chain_id])}",
            "timestamp": now,
            "actor": actor,
            "action": action,
            "evidence_ref": evidence_ref,
            "description": description,
            "prev_custody_hash": prev_hash,
            "signature": self._sign_event({
                "custody_id": None,  # Exclude for signing
                "timestamp": now,
                "actor": actor,
                "action": action,
                "evidence_ref": evidence_ref,
                "description": description,
                "prev_custody_hash": prev_hash
            })
        }
        event["chain_hash"] = self._compute_chain_hash(self.active_chains[chain_id] + [event])
        
        self.active_chains[chain_id].append(event)
        
        # Link to ledger if available
        if self.ledger:
            await self.ledger.append_event({
                "action": "custody_build",
                "chain_id": chain_id,
                "custody_event": event["custody_id"]
            })
        
        # Log to memory
        await self.mem.log_command(event)
        
        self.logger.info(f"Built custody event {event['custody_id']} for chain {chain_id}")
        
        return event

    async def finalize_chain(self, chain_id: str, persist_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Finalize a chain by persisting it and verifying integrity.
        
        :param chain_id: ID of the chain to finalize
        :param persist_file: Optional JSON file path for persistence
        :return: Chain summary with verification status
        """
        if chain_id not in self.active_chains:
            raise ValueError(f"Chain {chain_id} not found")
        
        chain = self.active_chains[chain_id]
        summary = {
            "chain_id": chain_id,
            "total_events": len(chain),
            "start_timestamp": chain[0]["timestamp"],
            "end_timestamp": chain[-1]["timestamp"],
            "final_hash": chain[-1]["chain_hash"]
        }
        
        # Verify all signatures
        valid_sigs = all(self._verify_signature(e, e["signature"]) for e in chain)
        summary["signature_valid"] = valid_sigs
        
        # Persist if requested
        if persist_file:
            with open(persist_file, 'w') as f:
                json.dump({"chain": chain, "summary": summary}, f, indent=2)
            self.logger.info(f"Persisted chain {chain_id} to {persist_file}")
        
        # Clear active after finalize
        del self.active_chains[chain_id]
        
        self.logger.info(f"Finalized chain {chain_id}: {summary}")
        return summary

    async def verify_chain(self, chain_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify an existing chain's integrity and signatures.
        
        :param chain_data: Loaded chain dict with "chain" key
        :return: Verification report
        """
        chain = chain_data.get("chain", [])
        report = {
            "chain_id": chain_data.get("chain_id"),
            "total_events": len(chain),
            "hash_valid": True,
            "signatures_valid": True,
            "issues": []
        }
        
        prev_hash = "genesis"
        for i, event in enumerate(chain):
            # Check chain hash continuity
            if event["prev_custody_hash"] != prev_hash:
                report["hash_valid"] = False
                report["issues"].append(f"Hash mismatch at event {i}: expected {prev_hash}, got {event['prev_custody_hash']}")
            
            # Verify signature
            if not self._verify_signature(event, event["signature"]):
                report["signatures_valid"] = False
                report["issues"].append(f"Invalid signature at event {i}: {event['custody_id']}")
            
            prev_hash = event["chain_hash"]  # Note: Use chain_hash for prev in cumulative setup
        
        # Final hash recompute
        recomputed_hash = self._compute_chain_hash(chain)
        if recomputed_hash != chain[-1]["chain_hash"] if chain else True:
            report["hash_valid"] = False
            report["issues"].append("Final chain hash mismatch")
        
        self.logger.info(f"Verified chain {report['chain_id']}: {'Valid' if report['hash_valid'] and report['signatures_valid'] else 'Issues'} ({len(report['issues'])})")
        return report

# Example usage / integration (e.g., post-WormGPT forensics)
async def example_custody_build():
    """Demo: Build a chain of custody for a forensic analysis."""
    builder = ChainOfCustodyBuilder()
    
    # Simulate evidence from ledger
    evidence_ref = "evt_1732999999_1"  # Mock ledger event
    
    # Build events
    await builder.build_custody_event(
        actor="forensic_analyst_001",
        action="acquired",
        evidence_ref=evidence_ref,
        description="Initial capture of adversarial prompt.",
        chain_id="analysis_001"
    )
    
    await builder.build_custody_event(
        actor="ai_detector_system",
        action="analyzed",
        evidence_ref=evidence_ref,
        description="WormGPT detection: critical risk, policy puppetry flagged.",
        chain_id="analysis_001"
    )
    
    await builder.build_custody_event(
        actor="security_officer_007",
        action="reviewed",
        evidence_ref=evidence_ref,
        description="Manual review and quarantine approved.",
        chain_id="analysis_001"
    )
    
    # Finalize
    summary = await builder.finalize_chain("analysis_001", persist_file="custody_chain_001.json")
    print(json.dumps(summary, indent=2))
    
    # Verify
    with open("custody_chain_001.json", 'r') as f:
        chain_data = json.load(f)
    report = await builder.verify_chain(chain_data)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    import os  # For key file check
    asyncio.run(example_custody_build())
