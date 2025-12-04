"""
core/adapters/proxy_adapter.py

A simple proxy adapter for handling external service calls with proxy support.
This adapter can be used to route requests through a proxy server for HTTP/HTTPS traffic.
Extended to support async operations using aiohttp.ClientSession for lightweight, async-friendly HTTP client.
Supports LLM API calls by providing a configurable async session.
"""

import aiohttp
import asyncio
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod


class BaseProxyAdapter(ABC):
    """
    Abstract base class for proxy adapters.
    """

    @abstractmethod
    def get_proxy_config(self) -> Optional[Dict[str, str]]:
        """
        Returns the proxy configuration dictionary.
        """
        pass

    def get_async_session(self) -> aiohttp.ClientSession:
        """
        Returns a aiohttp.ClientSession configured with the proxy settings.
        This can be used for async LLM clients or other libraries that accept a custom async session.
        Note: Ensure proper cleanup by using 'async with' context manager.
        """
        connector = aiohttp.TCPConnector()
        proxies = self.get_proxy_config()
        return aiohttp.ClientSession(connector=connector, trust_env=True, proxy=proxies)

    async def make_request(self, method: str, url: str, **kwargs) -> aiohttp.ClientResponse:
        """
        Makes an asynchronous HTTP request using a session configured with the proxy.
        """
        async with self.get_async_session() as session:
            async with session.request(method, url, **kwargs) as response:
                return response


class HttpProxyAdapter(BaseProxyAdapter):
    """
    Adapter for HTTP proxy.
    """

    def __init__(self, proxy_url: str, username: Optional[str] = None, password: Optional[str] = None):
        self.proxy_url = proxy_url
        self.username = username
        self.password = password

    def get_proxy_config(self) -> Optional[str]:
        """
        For aiohttp, proxy config is a single string or None (per-session proxy).
        Returns the proxy URL string, or None if no proxy.
        """
        if self.username and self.password:
            auth = f"{self.username}:{self.password}@"
        else:
            auth = ""
        return f"http://{auth}{self.proxy_url}"


class SocksProxyAdapter(BaseProxyAdapter):
    """
    Adapter for SOCKS proxy (requires 'aiohttp-socks' installed).
    Note: aiohttp requires additional library for SOCKS support; standard aiohttp uses HTTP proxies.
    """

    def __init__(self, proxy_url: str, username: Optional[str] = None, password: Optional[str] = None):
        self.proxy_url = proxy_url
        self.username = username
        self.password = password

    def get_proxy_config(self) -> Optional[str]:
        """
        For aiohttp with aiohttp-socks, proxy config is a single string.
        Returns the SOCKS proxy URL string, or None if no proxy.
        """
        if self.username and self.password:
            auth = f"{self.username}:{self.password}@"
        else:
            auth = ""
        return f"socks5://{auth}{self.proxy_url}"


# Example usage
async def main():
    # Example with HTTP proxy
    adapter = HttpProxyAdapter("proxy.example.com:8080", "user", "pass")
    
    # Simple async request
    response = await adapter.make_request("GET", "https://httpbin.org/ip")
    print("IP via proxy:", await response.json())
    
    # For LLM support: Get an async session to pass to async LLM clients
    # Example (uncomment and configure if using openai or similar; may require adapter for aiohttp):
    # async with adapter.get_async_session() as session:
    #     # Custom transport or adapter might be needed for OpenAI
    #     # openai_client = openai.AsyncOpenAI(http_client=AsyncHTTPClient(session=session))  # Hypothetical
    #     # response = await openai_client.chat.completions.create(...)

if __name__ == "__main__":
    asyncio.run(main())
