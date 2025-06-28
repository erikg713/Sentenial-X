#!/usr/bin/env python3
"""
Sentenial-X Professional Reconnaissance Module

A comprehensive, modular reconnaissance framework for domain intelligence gathering.
Designed for professional offensive and defensive security operations.

Author: Sentenial-X Security Team
License: MIT
Version: 3.0.0
"""

import argparse
import json
import logging
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# === Configuration ===
USER_AGENT = "Sentenial-X-Recon/3.0 (+https://github.com/erikg713/Sentenial-X-A.I.)"
DEFAULT_TIMEOUT = 10
DEFAULT_HEADERS = {"User-Agent": USER_AGENT}
COMMON_PORTS = [21, 22, 23, 25, 53, 80, 110, 143, 443, 445, 993, 995, 3389, 5432, 3306]
MAX_RETRIES = 3
BACKOFF_FACTOR = 1.0
RETRY_STATUS_CODES = [500, 502, 503, 504, 520, 521, 522, 523, 524]


# === Logging Configuration ===
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Configure comprehensive logging for reconnaissance operations.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("sentenial_recon")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    if not logger.handlers:
        # Console handler for user feedback
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)
        console_format = logging.Formatter('[%(levelname)s] %(message)s')
        console_handler.setFormatter(console_format)
        
        # File handler for detailed audit trail
        file_handler = logging.FileHandler('sentenial_recon.log', mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_format)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger


# === Global logger instance ===
logger = setup_logging()


# === Utility Functions ===
def retry_on_failure(
    max_attempts: int = MAX_RETRIES,
    delay: float = 1.0,
    backoff: float = BACKOFF_FACTOR,
    exceptions: Tuple = (Exception,)
) -> callable:
    """
    Decorator for implementing retry logic with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each failed attempt
        exceptions: Tuple of exceptions to catch and retry on
        
    Returns:
        Decorated function with retry capability
    """
    def decorator(func: callable) -> callable:
        def wrapper(*args, **kwargs):
            attempt = 0
            current_delay = delay
            
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts: {e}")
                        raise
                    
                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt}/{max_attempts}), "
                        f"retrying in {current_delay:.1f}s: {e}"
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
            
        return wrapper
    return decorator


def create_session(proxies: Optional[Dict[str, str]] = None) -> requests.Session:
    """
    Create a configured requests session with retry strategy.
    
    Args:
        proxies: Optional proxy configuration
        
    Returns:
        Configured requests session
    """
    session = requests.Session()
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=MAX_RETRIES,
        status_forcelist=RETRY_STATUS_CODES,
        backoff_factor=BACKOFF_FACTOR,
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    if proxies:
        session.proxies.update(proxies)
        logger.info(f"Configured session with proxies: {proxies}")
    
    return session


def sanitize_domain(domain: str) -> str:
    """
    Sanitize and validate domain input.
    
    Args:
        domain: Raw domain input
        
    Returns:
        Sanitized domain string
        
    Raises:
        ValueError: If domain is invalid
    """
    # Remove protocol if present
    if "://" in domain:
        domain = urlparse(domain).netloc or urlparse(domain).path
    
    # Remove port if present
    domain = domain.split(':')[0]
    
    # Basic domain validation - allow localhost and IP addresses
    domain = domain.lower().strip()
    if not domain:
        raise ValueError("Domain cannot be empty")
    
    # Allow localhost, IP addresses, and standard domains
    if not (domain == "localhost" or 
            re.match(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', domain) or
            re.match(r'^(\d{1,3}\.){3}\d{1,3}$', domain)):
        raise ValueError(f"Invalid domain format: {domain}")
    
    return domain


# === Core Reconnaissance Modules ===
class IPResolver:
    """Handle IP address resolution with comprehensive error handling."""
    
    @staticmethod
    @retry_on_failure(exceptions=(socket.gaierror, socket.timeout))
    def resolve_ipv4(domain: str) -> Optional[str]:
        """
        Resolve domain to IPv4 address.
        
        Args:
            domain: Target domain name
            
        Returns:
            IPv4 address string or None if resolution fails
        """
        try:
            ip = socket.gethostbyname(domain)
            logger.info(f"Resolved IPv4 for {domain}: {ip}")
            return ip
        except (socket.gaierror, socket.timeout) as e:
            logger.error(f"IPv4 resolution failed for {domain}: {e}")
            return None
    
    @staticmethod
    @retry_on_failure(exceptions=(socket.gaierror, socket.timeout))
    def resolve_ipv6(domain: str) -> Optional[str]:
        """
        Resolve domain to IPv6 address.
        
        Args:
            domain: Target domain name
            
        Returns:
            IPv6 address string or None if resolution fails
        """
        try:
            result = socket.getaddrinfo(domain, None, socket.AF_INET6)
            if result:
                ipv6 = result[0][4][0]
                logger.info(f"Resolved IPv6 for {domain}: {ipv6}")
                return ipv6
        except (socket.gaierror, socket.timeout) as e:
            logger.debug(f"IPv6 resolution failed for {domain}: {e}")
        return None


class HTTPHeaderAnalyzer:
    """Analyze HTTP headers and detect technology stacks."""
    
    def __init__(self, session: requests.Session, custom_headers: Optional[Dict[str, str]] = None):
        """
        Initialize header analyzer.
        
        Args:
            session: Configured requests session
            custom_headers: Optional custom headers to include in requests
        """
        self.session = session
        self.headers = DEFAULT_HEADERS.copy()
        if custom_headers:
            self.headers.update(custom_headers)
    
    def fetch_headers(self, url: str, timeout: int = DEFAULT_TIMEOUT) -> Dict[str, str]:
        """
        Fetch HTTP response headers with error handling.
        
        Args:
            url: Target URL
            timeout: Request timeout in seconds
            
        Returns:
            Dictionary of response headers
        """
        try:
            logger.debug(f"Fetching headers from: {url}")
            response = self.session.head(
                url, 
                headers=self.headers, 
                timeout=timeout,
                allow_redirects=True,
                verify=True
            )
            
            headers = dict(response.headers)
            logger.info(f"Successfully fetched headers from {url} (Status: {response.status_code})")
            return headers
            
        except requests.exceptions.SSLError:
            logger.warning(f"SSL verification failed for {url}, trying without verification")
            try:
                response = self.session.head(
                    url, 
                    headers=self.headers, 
                    timeout=timeout,
                    allow_redirects=True,
                    verify=False
                )
                headers = dict(response.headers)
                logger.info(f"Successfully fetched headers from {url} without SSL verification")
                return headers
            except Exception as e:
                logger.error(f"Header fetch failed for {url}: {e}")
                return {}
        except Exception as e:
            logger.error(f"Header fetch failed for {url}: {e}")
            return {}
    
    def detect_technology_stack(self, headers: Dict[str, str]) -> List[str]:
        """
        Detect technology stack from HTTP headers.
        
        Args:
            headers: HTTP response headers
            
        Returns:
            List of detected technologies
        """
        technologies = []
        
        # Server header analysis
        server = headers.get('server', '').lower()
        if server:
            technologies.append(f"Server: {headers['server']}")
        
        # X-Powered-By header
        powered_by = headers.get('x-powered-by', '')
        if powered_by:
            technologies.append(f"X-Powered-By: {powered_by}")
        
        # Content Management Systems
        generator = headers.get('x-generator', '')
        if generator:
            technologies.append(f"Generator: {generator}")
        
        # Web Application Frameworks
        framework_headers = [
            'x-aspnet-version', 'x-django-version', 'x-rails-version',
            'x-drupal-cache', 'x-pingback', 'x-wp-super-cache'
        ]
        
        for header in framework_headers:
            value = headers.get(header, '')
            if value:
                technologies.append(f"{header}: {value}")
        
        # Security headers detection
        security_headers = [
            'x-frame-options', 'x-xss-protection', 'x-content-type-options',
            'strict-transport-security', 'content-security-policy'
        ]
        
        for header in security_headers:
            if header in headers:
                technologies.append(f"Security: {header}")
        
        # CDN Detection
        cdn_headers = ['cf-ray', 'x-cache', 'x-served-by', 'x-amz-cf-id']
        for header in cdn_headers:
            if header in headers:
                technologies.append(f"CDN: {header}")
        
        logger.info(f"Detected technologies: {technologies}")
        return sorted(list(set(technologies)))  # Remove duplicates and sort


class WHOISAnalyzer:
    """Handle WHOIS lookups and data extraction."""
    
    @staticmethod
    @retry_on_failure(exceptions=(subprocess.SubprocessError, subprocess.TimeoutExpired))
    def perform_lookup(domain: str, timeout: int = 30) -> str:
        """
        Perform WHOIS lookup for domain.
        
        Args:
            domain: Target domain
            timeout: Lookup timeout in seconds
            
        Returns:
            WHOIS data as string
        """
        try:
            logger.debug(f"Performing WHOIS lookup for: {domain}")
            result = subprocess.run(
                ["whois", domain],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False
            )
            
            if result.returncode == 0 and result.stdout:
                logger.info(f"WHOIS lookup successful for {domain}")
                return result.stdout
            else:
                logger.warning(f"WHOIS lookup returned no data for {domain}")
                return "WHOIS lookup returned no data"
                
        except subprocess.TimeoutExpired:
            logger.error(f"WHOIS lookup timed out for {domain}")
            return "WHOIS lookup timed out"
        except FileNotFoundError:
            logger.error("WHOIS command not found - please install whois package")
            return "WHOIS command not available"
        except Exception as e:
            logger.error(f"WHOIS lookup failed for {domain}: {e}")
            return f"WHOIS lookup failed: {str(e)}"
    
    @staticmethod
    def extract_emails(whois_data: str) -> List[str]:
        """
        Extract email addresses from WHOIS data.
        
        Args:
            whois_data: Raw WHOIS data
            
        Returns:
            List of unique email addresses
        """
        # Improved email regex pattern
        email_pattern = r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
        emails = re.findall(email_pattern, whois_data, re.IGNORECASE)
        
        # Filter out common privacy/placeholder emails
        filtered_emails = []
        privacy_domains = ['whoisguard', 'privacyguardian', 'domainsbyproxy', 'contactprivacy']
        
        for email in emails:
            email_lower = email.lower()
            if not any(domain in email_lower for domain in privacy_domains):
                filtered_emails.append(email)
        
        unique_emails = sorted(list(set(filtered_emails)))
        logger.info(f"Extracted {len(unique_emails)} unique emails from WHOIS data")
        return unique_emails


class PortScanner:
    """Network port scanning functionality."""
    
    @staticmethod
    def scan_port(ip: str, port: int, timeout: float = 1.0) -> bool:
        """
        Check if a specific port is open.
        
        Args:
            ip: Target IP address
            port: Port number to check
            timeout: Connection timeout
            
        Returns:
            True if port is open, False otherwise
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((ip, port))
            sock.close()
            return result == 0
        except Exception:
            return False
    
    def scan_common_ports(
        self, 
        ip: str, 
        ports: List[int] = None, 
        timeout: float = 1.0
    ) -> Dict[int, str]:
        """
        Scan common ports on target IP.
        
        Args:
            ip: Target IP address
            ports: List of ports to scan (defaults to COMMON_PORTS)
            timeout: Connection timeout per port
            
        Returns:
            Dictionary mapping open ports to their status
        """
        if ports is None:
            ports = COMMON_PORTS
        
        open_ports = {}
        logger.info(f"Scanning {len(ports)} ports on {ip}")
        
        for port in ports:
            if self.scan_port(ip, port, timeout):
                service = self._identify_service(port)
                open_ports[port] = service
                logger.debug(f"Port {port} ({service}) is open on {ip}")
        
        logger.info(f"Found {len(open_ports)} open ports on {ip}")
        return open_ports
    
    @staticmethod
    def _identify_service(port: int) -> str:
        """
        Identify common services by port number.
        
        Args:
            port: Port number
            
        Returns:
            Service name or 'unknown'
        """
        service_map = {
            21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP", 53: "DNS",
            80: "HTTP", 110: "POP3", 143: "IMAP", 443: "HTTPS", 445: "SMB",
            993: "IMAPS", 995: "POP3S", 3389: "RDP", 5432: "PostgreSQL", 3306: "MySQL"
        }
        return service_map.get(port, "unknown")


# === Main Reconnaissance Engine ===
class ReconEngine:
    """Main reconnaissance engine orchestrating all analysis modules."""
    
    def __init__(
        self, 
        proxies: Optional[Dict[str, str]] = None,
        custom_headers: Optional[Dict[str, str]] = None,
        user_agent: Optional[str] = None
    ):
        """
        Initialize reconnaissance engine.
        
        Args:
            proxies: Optional proxy configuration
            custom_headers: Optional custom HTTP headers
            user_agent: Optional custom user agent
        """
        self.session = create_session(proxies)
        
        headers = custom_headers.copy() if custom_headers else {}
        if user_agent:
            headers['User-Agent'] = user_agent
        
        self.ip_resolver = IPResolver()
        self.header_analyzer = HTTPHeaderAnalyzer(self.session, headers)
        self.whois_analyzer = WHOISAnalyzer()
        self.port_scanner = PortScanner()
        
        logger.info("Reconnaissance engine initialized")
    
    def perform_comprehensive_recon(
        self, 
        domain: str,
        enable_port_scan: bool = True,
        port_scan_timeout: float = 1.0
    ) -> Dict[str, Any]:
        """
        Perform comprehensive reconnaissance on target domain.
        
        Args:
            domain: Target domain name
            enable_port_scan: Whether to perform port scanning
            port_scan_timeout: Timeout for port scanning operations
            
        Returns:
            Comprehensive reconnaissance results
        """
        start_time = time.time()
        domain = sanitize_domain(domain)
        
        logger.info(f"Starting comprehensive reconnaissance for: {domain}")
        
        # Initialize result structure
        result = {
            "domain": domain,
            "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            "scan_duration": 0.0,
            "ip_addresses": {
                "ipv4": None,
                "ipv6": None
            },
            "http_analysis": {
                "https_headers": {},
                "http_headers": {},
                "technology_stack": []
            },
            "whois_data": {
                "raw": "",
                "emails": []
            },
            "network_scan": {
                "open_ports": {},
                "scan_enabled": enable_port_scan
            },
            "errors": []
        }
        
        # Phase 1: IP Resolution
        try:
            result["ip_addresses"]["ipv4"] = self.ip_resolver.resolve_ipv4(domain)
            result["ip_addresses"]["ipv6"] = self.ip_resolver.resolve_ipv6(domain)
        except Exception as e:
            error_msg = f"IP resolution error: {str(e)}"
            result["errors"].append(error_msg)
            logger.error(error_msg)
        
        # Phase 2: HTTP Header Analysis with HTTPS/HTTP fallback
        try:
            # Try HTTPS first
            https_url = f"https://{domain}"
            https_headers = self.header_analyzer.fetch_headers(https_url)
            result["http_analysis"]["https_headers"] = https_headers
            
            if https_headers:
                result["http_analysis"]["technology_stack"] = \
                    self.header_analyzer.detect_technology_stack(https_headers)
            else:
                # Fallback to HTTP
                http_url = f"http://{domain}"
                http_headers = self.header_analyzer.fetch_headers(http_url)
                result["http_analysis"]["http_headers"] = http_headers
                
                if http_headers:
                    result["http_analysis"]["technology_stack"] = \
                        self.header_analyzer.detect_technology_stack(http_headers)
        except Exception as e:
            error_msg = f"HTTP analysis error: {str(e)}"
            result["errors"].append(error_msg)
            logger.error(error_msg)
        
        # Phase 3: WHOIS Analysis
        try:
            whois_data = self.whois_analyzer.perform_lookup(domain)
            result["whois_data"]["raw"] = whois_data
            result["whois_data"]["emails"] = self.whois_analyzer.extract_emails(whois_data)
        except Exception as e:
            error_msg = f"WHOIS analysis error: {str(e)}"
            result["errors"].append(error_msg)
            logger.error(error_msg)
        
        # Phase 4: Port Scanning (if enabled and IP available)
        if enable_port_scan and result["ip_addresses"]["ipv4"]:
            try:
                open_ports = self.port_scanner.scan_common_ports(
                    result["ip_addresses"]["ipv4"], 
                    timeout=port_scan_timeout
                )
                result["network_scan"]["open_ports"] = open_ports
            except Exception as e:
                error_msg = f"Port scanning error: {str(e)}"
                result["errors"].append(error_msg)
                logger.error(error_msg)
        
        # Calculate scan duration
        result["scan_duration"] = round(time.time() - start_time, 2)
        
        logger.info(f"Reconnaissance completed for {domain} in {result['scan_duration']}s")
        return result


# === Command Line Interface ===
def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure argument parser for CLI interface.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Sentenial-X Professional Reconnaissance Tool",
        epilog="""
Examples:
  %(prog)s example.com
  %(prog)s example.com -v --output results.json
  %(prog)s example.com --proxy http://127.0.0.1:8080 --no-scan
  %(prog)s example.com --header "Authorization:Bearer token" --header "X-Custom:value"
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Positional arguments
    parser.add_argument(
        "domain",
        help="Target domain for reconnaissance"
    )
    
    # Optional arguments
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output with full JSON results"
    )
    
    parser.add_argument(
        "--output",
        metavar="FILE",
        help="Save JSON results to specified file"
    )
    
    parser.add_argument(
        "--proxy",
        metavar="URL",
        help="HTTP/HTTPS proxy URL (e.g., http://127.0.0.1:8080)"
    )
    
    parser.add_argument(
        "--no-scan",
        action="store_true",
        help="Disable port scanning (faster reconnaissance)"
    )
    
    parser.add_argument(
        "--header",
        action="append",
        metavar="KEY:VALUE",
        help="Add custom HTTP header (can be used multiple times)"
    )
    
    parser.add_argument(
        "--user-agent",
        metavar="STRING",
        help="Custom User-Agent string"
    )
    
    parser.add_argument(
        "--timeout",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help="Port scan timeout in seconds (default: 1.0)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )
    
    return parser


def parse_custom_headers(header_list: List[str]) -> Dict[str, str]:
    """
    Parse custom headers from command line arguments.
    
    Args:
        header_list: List of header strings in "Key:Value" format
        
    Returns:
        Dictionary of parsed headers
    """
    headers = {}
    for header in header_list or []:
        if ":" in header:
            key, value = header.split(":", 1)
            headers[key.strip()] = value.strip()
        else:
            logger.warning(f"Invalid header format ignored: {header}")
    return headers


def format_output(result: Dict[str, Any], verbose: bool = False) -> str:
    """
    Format reconnaissance results for display.
    
    Args:
        result: Reconnaissance results dictionary
        verbose: Whether to show verbose output
        
    Returns:
        Formatted output string
    """
    if verbose:
        return json.dumps(result, indent=2, ensure_ascii=False)
    
    # Compact summary format
    lines = [
        f"Domain: {result['domain']}",
        f"Timestamp: {result['timestamp']}",
        f"Scan Duration: {result['scan_duration']}s",
        "",
        "IP Addresses:"
    ]
    
    if result['ip_addresses']['ipv4']:
        lines.append(f"  IPv4: {result['ip_addresses']['ipv4']}")
    if result['ip_addresses']['ipv6']:
        lines.append(f"  IPv6: {result['ip_addresses']['ipv6']}")
    
    if result['http_analysis']['technology_stack']:
        lines.append("\nTechnology Stack:")
        for tech in result['http_analysis']['technology_stack']:
            lines.append(f"  • {tech}")
    
    if result['whois_data']['emails']:
        lines.append("\nEmail Addresses:")
        for email in result['whois_data']['emails']:
            lines.append(f"  • {email}")
    
    if result['network_scan']['open_ports']:
        lines.append("\nOpen Ports:")
        for port, service in result['network_scan']['open_ports'].items():
            lines.append(f"  • {port} ({service})")
    
    if result['errors']:
        lines.append("\nErrors:")
        for error in result['errors']:
            lines.append(f"  • {error}")
    
    return "\n".join(lines)


def main() -> int:
    """
    Main entry point for CLI interface.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Configure logging
    global logger
    logger = setup_logging(args.log_level)
    
    try:
        # Parse proxy configuration
        proxies = None
        if args.proxy:
            proxies = {"http": args.proxy, "https": args.proxy}
        
        # Parse custom headers
        custom_headers = parse_custom_headers(args.header)
        
        # Initialize reconnaissance engine
        engine = ReconEngine(
            proxies=proxies,
            custom_headers=custom_headers,
            user_agent=args.user_agent
        )
        
        # Perform reconnaissance
        result = engine.perform_comprehensive_recon(
            domain=args.domain,
            enable_port_scan=not args.no_scan,
            port_scan_timeout=args.timeout
        )
        
        # Output results
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"[+] Results saved to: {args.output}")
        
        # Display results
        output = format_output(result, args.verbose)
        print(output)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n[!] Reconnaissance interrupted by user")
        return 130
    except ValueError as e:
        logger.error(f"Input validation error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())