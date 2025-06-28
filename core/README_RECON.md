# Sentenial-X Professional Reconnaissance Module

A comprehensive, modular reconnaissance framework for domain intelligence gathering, designed for professional offensive and defensive security operations.

## Features

### Core Capabilities
- **Modular Architecture**: Extensible design with clear separation of concerns
- **Comprehensive Error Handling**: Robust error handling with detailed logging
- **Retry Logic**: Automatic retry with exponential backoff for network operations
- **HTTPS/HTTP Fallback**: Automatic fallback from HTTPS to HTTP for header analysis
- **Proxy Support**: HTTP/HTTPS proxy support for all network operations
- **Custom Headers**: Support for custom HTTP headers and user agents

### Intelligence Gathering
- **IP Resolution**: IPv4 and IPv6 address resolution
- **HTTP Header Analysis**: Technology stack detection from HTTP headers
- **WHOIS Analysis**: Domain registration information with email extraction
- **Port Scanning**: Network port scanning with service identification
- **Email Extraction**: Intelligent email extraction with privacy filter

### Output & Interface
- **Professional CLI**: Full-featured command-line interface with argparse
- **JSON Output**: Structured JSON output with UTC timestamps
- **Flexible Formatting**: Both verbose and compact output modes
- **File Export**: Save results to JSON files

## Installation

```bash
# Clone the repository
git clone https://github.com/erikg713/Sentenial-X-A.I.git
cd Sentenial-X-A.I

# Install Python dependencies
pip install requests urllib3

# Install system dependencies (optional, for WHOIS)
sudo apt-get install whois  # Ubuntu/Debian
# or
sudo yum install whois      # CentOS/RHEL
```

## Usage

### Basic Usage

```bash
# Basic reconnaissance
python3 core/recon.py example.com

# Verbose output with full JSON
python3 core/recon.py example.com --verbose

# Save results to file
python3 core/recon.py example.com --output results.json
```

### Advanced Usage

```bash
# Skip port scanning (faster)
python3 core/recon.py example.com --no-scan

# Use proxy
python3 core/recon.py example.com --proxy http://127.0.0.1:8080

# Custom headers
python3 core/recon.py example.com \
    --header "Authorization:Bearer token123" \
    --header "X-Custom:value"

# Custom user agent
python3 core/recon.py example.com --user-agent "MyTool/1.0"

# Adjust port scan timeout
python3 core/recon.py example.com --timeout 2.0

# Debug logging
python3 core/recon.py example.com --log-level DEBUG
```

### Command Line Options

```
usage: recon.py [-h] [-v] [--output FILE] [--proxy URL] [--no-scan] 
                [--header KEY:VALUE] [--user-agent STRING] [--timeout SECONDS]
                [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}] domain

positional arguments:
  domain                Target domain for reconnaissance

options:
  -h, --help           show this help message and exit
  -v, --verbose        Enable verbose output with full JSON results
  --output FILE        Save JSON results to specified file
  --proxy URL          HTTP/HTTPS proxy URL (e.g., http://127.0.0.1:8080)
  --no-scan           Disable port scanning (faster reconnaissance)
  --header KEY:VALUE   Add custom HTTP header (can be used multiple times)
  --user-agent STRING  Custom User-Agent string
  --timeout SECONDS    Port scan timeout in seconds (default: 1.0)
  --log-level LEVEL    Set logging level (default: INFO)
```

## Output Format

### JSON Structure

```json
{
  "domain": "example.com",
  "timestamp": "2025-06-28T17:41:29.885124Z",
  "scan_duration": 1.23,
  "ip_addresses": {
    "ipv4": "192.168.1.1",
    "ipv6": "2001:db8::1"
  },
  "http_analysis": {
    "https_headers": {
      "server": "nginx/1.18.0",
      "x-powered-by": "PHP/7.4.3"
    },
    "http_headers": {},
    "technology_stack": [
      "Server: nginx/1.18.0",
      "X-Powered-By: PHP/7.4.3"
    ]
  },
  "whois_data": {
    "raw": "Domain registration data...",
    "emails": [
      "admin@example.com",
      "tech@example.com"
    ]
  },
  "network_scan": {
    "open_ports": {
      "80": "HTTP",
      "443": "HTTPS",
      "22": "SSH"
    },
    "scan_enabled": true
  },
  "errors": []
}
```

### Compact Output

```
Domain: example.com
Timestamp: 2025-06-28T17:41:29.885124Z
Scan Duration: 1.23s

IP Addresses:
  IPv4: 192.168.1.1
  IPv6: 2001:db8::1

Technology Stack:
  • Server: nginx/1.18.0
  • X-Powered-By: PHP/7.4.3

Email Addresses:
  • admin@example.com
  • tech@example.com

Open Ports:
  • 80 (HTTP)
  • 443 (HTTPS)
  • 22 (SSH)
```

## API Usage

### Programmatic Usage

```python
from core.recon import ReconEngine

# Initialize engine
engine = ReconEngine(
    proxies={"http": "http://proxy:8080", "https": "http://proxy:8080"},
    custom_headers={"X-API-Key": "your-key"},
    user_agent="MyTool/1.0"
)

# Perform reconnaissance
result = engine.perform_comprehensive_recon(
    domain="example.com",
    enable_port_scan=True,
    port_scan_timeout=2.0
)

# Access results
print(f"Domain: {result['domain']}")
print(f"IPv4: {result['ip_addresses']['ipv4']}")
print(f"Technologies: {result['http_analysis']['technology_stack']}")
```

### Individual Modules

```python
from core.recon import IPResolver, HTTPHeaderAnalyzer, PortScanner

# IP Resolution
resolver = IPResolver()
ipv4 = resolver.resolve_ipv4("example.com")
ipv6 = resolver.resolve_ipv6("example.com")

# Port Scanning
scanner = PortScanner()
open_ports = scanner.scan_common_ports("192.168.1.1")

# Header Analysis
import requests
session = requests.Session()
analyzer = HTTPHeaderAnalyzer(session)
headers = analyzer.fetch_headers("https://example.com")
technologies = analyzer.detect_technology_stack(headers)
```

## Configuration

### Default Settings

```python
USER_AGENT = "Sentenial-X-Recon/3.0 (+https://github.com/erikg713/Sentenial-X-A.I.)"
DEFAULT_TIMEOUT = 10
COMMON_PORTS = [21, 22, 23, 25, 53, 80, 110, 143, 443, 445, 993, 995, 3389, 5432, 3306]
MAX_RETRIES = 3
BACKOFF_FACTOR = 1.0
```

### Logging

The tool automatically creates detailed logs in `sentenial_recon.log` with the following levels:
- **DEBUG**: Detailed debugging information
- **INFO**: General operational information
- **WARNING**: Warning messages for non-critical issues
- **ERROR**: Error messages for failed operations
- **CRITICAL**: Critical errors that may halt operation

## Security Considerations

### Operational Security
- Use proxies when conducting reconnaissance to maintain anonymity
- Be aware of rate limiting and detection mechanisms on target systems
- Respect robots.txt and terms of service when applicable
- Use appropriate timeouts to avoid overwhelming target systems

### Data Handling
- WHOIS data may contain personal information - handle responsibly
- Email addresses are filtered to remove privacy service placeholders
- All network operations include proper error handling and retry logic
- Logs may contain sensitive information - secure appropriately

## Testing

Run the comprehensive test suite:

```bash
python3 tests/test_recon.py
```

The test suite covers:
- Domain sanitization and validation
- IP resolution functionality
- Port scanning operations
- HTTP header analysis
- WHOIS data extraction
- CLI interface functionality
- JSON output formatting
- Integration workflows

## Architecture

### Modular Design

```
core/recon.py
├── IPResolver          # IP address resolution
├── HTTPHeaderAnalyzer  # HTTP header analysis and tech stack detection
├── WHOISAnalyzer      # WHOIS data retrieval and email extraction
├── PortScanner        # Network port scanning
└── ReconEngine        # Main orchestration engine
```

### Extensibility

The modular architecture allows for easy extension:

1. **New Analysis Modules**: Implement new analyzers following the existing patterns
2. **Custom Plugins**: Add specialized reconnaissance functionality
3. **Output Formats**: Extend the output formatting system
4. **Transport Protocols**: Add support for additional network protocols

## Contributing

1. Follow PEP 8 style guidelines
2. Add comprehensive docstrings with type hints
3. Include unit tests for new functionality
4. Update documentation for new features
5. Ensure backward compatibility

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Changelog

### Version 3.0.0
- Complete rewrite with modular architecture
- Added comprehensive error handling and logging
- Implemented retry logic with exponential backoff
- Added HTTPS/HTTP fallback functionality
- Enhanced CLI interface with all required options
- Added comprehensive test suite
- Improved email extraction with privacy filtering
- Added technology stack detection
- Professional code quality and documentation