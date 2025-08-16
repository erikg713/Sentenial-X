"""
Sentenial-X Reporting - upload.py
---------------------------------
Handles secure uploading of reports, logs, and forensic data
to multiple destinations (local, remote API, cloud storage).

Optimized for production with async uploads, retries, and structured logging.
"""

import os
import aiohttp
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from aiofiles import open as aio_open

# Configure structured logging
logger = logging.getLogger("sentenial.reporting.upload")
logger.setLevel(logging.INFO)


class ReportUploader:
    def __init__(self, upload_dir: str = "reports/output", retries: int = 3, timeout: int = 30):
        self.upload_dir = Path(upload_dir)
        self.retries = retries
        self.timeout = timeout
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    async def save_local(self, filename: str, content: str) -> Path:
        """Save report locally in the designated reports directory."""
        path = self.upload_dir / filename
        try:
            async with aio_open(path, "w") as f:
                await f.write(content)
            logger.info(f"[UPLOAD] Report saved locally: {path}")
            return path
        except Exception as e:
            logger.error(f"[UPLOAD ERROR] Failed to save report locally: {e}")
            raise

    async def upload_remote(self, url: str, data: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> bool:
        """Upload report to a remote API endpoint with retries."""
        attempt = 0
        while attempt < self.retries:
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                    async with session.post(url, json=data, headers=headers) as resp:
                        if resp.status == 200:
                            logger.info(f"[UPLOAD] Report successfully uploaded to {url}")
                            return True
                        else:
                            logger.warning(f"[UPLOAD WARN] Upload failed (status {resp.status}): {await resp.text()}")
            except Exception as e:
                logger.error(f"[UPLOAD ERROR] Attempt {attempt+1} failed: {e}")
            
            attempt += 1
            await asyncio.sleep(2 ** attempt)  # exponential backoff
        
        logger.critical(f"[UPLOAD FAIL] Could not upload report after {self.retries} attempts")
        return False

    async def upload(self, filename: str, content: str, remote_url: Optional[str] = None,
                     headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Save report locally and optionally upload to a remote server.
        Returns dict with status information.
        """
        result = {"local_path": None, "remote_uploaded": False}

        # Save locally
        result["local_path"] = str(await self.save_local(filename, content))

        # Upload to remote if provided
        if remote_url:
            result["remote_uploaded"] = await self.upload_remote(remote_url, {"filename": filename, "content": content}, headers)

        return result


# Example usage
if __name__ == "__main__":
    async def main():
        uploader = ReportUploader()
        report_content = "Threat Analysis Report: All systems nominal."
        result = await uploader.upload("threat_report.txt", report_content, remote_url="https://example.com/upload")
        print(result)

    asyncio.run(main())
