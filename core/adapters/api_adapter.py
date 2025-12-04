"""
core/adapters/api_adapter.py

Sentenial-X Adapters API Adapter Module - provides a unified adapter for external API interactions,
handling authentication, rate limiting, and response parsing for services like fact-check APIs,
notification endpoints, or third-party forensics tools. Integrates with IncidentQueue for async
processing, ReflexManager for triggers, and forensics modules for auditing API calls.
Supports configurable endpoints and auto-retry on failures.
"""

import asyncio
import aiohttp
import time
import json
from typing import Dict, Any, Optional, List
from aiohttp import ClientSession, ClientTimeout
from cli.memory_adapter import get_adapter
from cli.logger import default_logger
from core.orchestrator.incident_queue import IncidentQueue
from core.orchestrator.incident_reflex_manager import IncidentReflexManager
from core.forensics.ledger_sequencer import LedgerSequencer
from core.forensics.chain_of_custody_builder import ChainOfCustodyBuilder

# Default API config schema
API_CONFIG_SCHEMA = {
    "endpoint": str,
    "method": str,  # GET, POST, etc.
    "headers": Optional[Dict[str, str]],
    "auth": Optional[Dict[str, str]],  # e.g., {"type": "bearer", "token": "..."}
    "params": Optional[Dict[str, Any]],
    "body": Optional[Dict[str, Any]],
    "timeout": float,  # Seconds
    "retries": int
}

class ApiAdapter:
    """
    Unified adapter for asynchronous API calls with rate limiting, retries, and forensics integration.
    
    :param queue: Optional IncidentQueue for post-call escalation
    :param reflex: Optional ReflexManager for API response triggers
    :param ledger: Optional LedgerSequencer for call logging
    :param custody: Optional ChainOfCustodyBuilder for evidence chains
    :param rate_limit: Calls per minute
    """
    def __init__(self, queue: Optional[IncidentQueue] = None,
                 reflex: Optional[IncidentReflexManager] = None,
                 ledger: Optional[LedgerSequencer] = None,
                 custody: Optional[ChainOfCustodyBuilder] = None,
                 rate_limit: int = 60):
        self.queue = queue or IncidentQueue()
        self.reflex = reflex or IncidentReflexManager(self.queue)
        self.ledger = ledger or LedgerSequencer()
        self.custody = custody or ChainOfCustodyBuilder(self.ledger)
        self.mem = get_adapter()
        self.logger = default_logger
        self.rate_limit = rate_limit
        self.last_call_time = 0.0
        self.session: Optional[ClientSession] = None

    async def _init_session(self):
        if not self.session:
            self.session = ClientSession(timeout=ClientTimeout(total=30))

    async def call_api(self, config: Dict[str, Any], payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make an asynchronous API call with config, handling auth, retries, and rate limiting.
        
        :param config: API config dict
        :param payload: Optional override for body/params
        :return: Response data with status
        """
        await self._init_session()
        
        # Rate limiting
        elapsed = time.time() - self.last_call_time
        if elapsed < 60 / self.rate_limit:
            await asyncio.sleep(60 / self.rate_limit - elapsed)
        self.last_call_time = time.time()
        
        method = config.get("method", "GET").upper()
        headers = config.get("headers", {})
        auth = config.get("auth")
        if auth and auth.get("type") == "bearer":
            headers["Authorization"] = f"Bearer {auth['token']}"
        
        params = payload.get("params") if payload else config.get("params", {})
        body = payload.get("body") if payload else config.get("body")
        
        retries = config.get("retries", 3)
        for attempt in range(retries):
            try:
                async with self.session.request(
                    method=method,
                    url=config["endpoint"],
                    headers=headers,
                    params=params,
                    json=body if method in ["POST", "PUT"] else None
                ) as resp:
                    status = resp.status
                    content = await resp.text()
                    data = json.loads(content) if "application/json" in resp.headers.get("Content-Type", "") else content
                    
                    # Log call
                    log_entry = {
                        "action": "api_call",
                        "endpoint": config["endpoint"],
                        "method": method,
                        "status": status,
                        "response": str(data)[:500] + "..." if len(str(data)) > 500 else data
                    }
                    ledger_event = await self.ledger.append_event(log_entry)
                    
                    # Build custody if error
                    if status >= 400:
                        await self.custody.build_custody_event(
                            actor="api_adapter",
                            action="api_error",
                            evidence_ref=ledger_event["event_id"],
                            description=f"API call failed: {status}",
                            chain_id=f"api_chain_{ledger_event['event_id']}"
                        )
                        # Enqueue incident
                        await self.queue.enqueue("high", log_entry)
                    
                    # Trigger reflex on success/failure
                    if status == 200:
                        await self.reflex.trigger_reflex("low", log_entry)  # Mock low for success
                    else:
                        await self.reflex.trigger_reflex("high", log_entry)
                    
                    self.logger.info(f"API call to {config['endpoint']}: {status}")
                    
                    return {"status": status, "data": data}
            except Exception as e:
                self.logger.error(f"API call attempt {attempt+1} failed: {str(e)}")
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

    async def batch_call(self, configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batch asynchronous API calls."""
        tasks = [self.call_api(cfg) for cfg in configs]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def close(self):
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()
        self.logger.info("API adapter session closed")

# Example usage / integration (e.g., fact-check API call)
async def example_api_call():
    """Demo: Call a mock API and handle response."""
    adapter = ApiAdapter()
    
    # Mock config (e.g., for fact-check API)
    config = {
        "endpoint": "https://api.factcheck.example/verify",
        "method": "POST",
        "headers": {"Content-Type": "application/json"},
        "auth": {"type": "bearer", "token": "mock_token"},
        "body": {"claim": "AI was invented in 1956."},
        "retries": 2
    }
    
    response = await adapter.call_api(config)
    print(json.dumps(response, indent=2))
    
    await adapter.close()

if __name__ == "__main__":
    asyncio.run(example_api_call())
