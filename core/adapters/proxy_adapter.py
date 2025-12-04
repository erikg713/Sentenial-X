"""
core/adapters/proxy_adapter.py

A simple proxy adapter for handling external service calls with proxy support.
This adapter can be used to route requests through a proxy server for HTTP/HTTPS traffic.
"""

import requests
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

    def make_request(self, method: str, url: str, **kwargs) -> requests.Response:
        """
        Makes an HTTP request using the proxy configuration.
        """
        proxies = self.get_proxy_config()
        if proxies:
            kwargs['proxies'] = proxies
        return requests.request(method, url, **kwargs)


class HttpProxyAdapter(BaseProxyAdapter):
    """
    Adapter for HTTP proxy.
    """

    def __init__(self, proxy_url: str, username: Optional[str] = None, password: Optional[str] = None):
        self.proxy_url = proxy_url
        self.username = username
        self.password = password

    def get_proxy_config(self) -> Optional[Dict[str, str]]:
        if self.username and self.password:
            auth = f"{self.username}:{self.password}@"
        else:
            auth = ""
        return {
            'http': f'http://{auth}{self.proxy_url}',
            'https': f'http://{auth}{self.proxy_url}'
        }


class SocksProxyAdapter(BaseProxyAdapter):
    """
    Adapter for SOCKS proxy (requires 'requests[socks]' installed).
    """

    def __init__(self, proxy_url: str, username: Optional[str] = None, password: Optional[str] = None):
        self.proxy_url = proxy_url
        self.username = username
        self.password = password

    def get_proxy_config(self) -> Optional[Dict[str, str]]:
        if self.username and self.password:
            auth = f"{self.username}:{self.password}@"
        else:
            auth = ""
        return {
            'http': f'socks5://{auth}{self.proxy_url}',
            'https': f'socks5://{auth}{self.proxy_url}'
        }


# Example usage
if __name__ == "__main__":
    # Example with HTTP proxy
    adapter = HttpProxyAdapter("proxy.example.com:8080", "user", "pass")
    response = adapter.make_request("GET", "https://httpbin.org/ip")
    print(response.json())
