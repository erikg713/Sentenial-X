#!/usr/bin/env python3
"""
Comprehensive test suite for the enhanced recon.py module.

This test suite validates all major functionality of the professional reconnaissance tool
including domain validation, header analysis, port scanning, and output formatting.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
import socket

# Add the core directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

import recon


class TestDomainSanitization(unittest.TestCase):
    """Test domain input sanitization and validation."""
    
    def test_valid_domains(self):
        """Test valid domain inputs."""
        test_cases = [
            ("example.com", "example.com"),
            ("EXAMPLE.COM", "example.com"),
            ("sub.example.com", "sub.example.com"),
            ("test-domain.co.uk", "test-domain.co.uk"),
            ("localhost", "localhost"),
            ("192.168.1.1", "192.168.1.1"),
        ]
        
        for input_domain, expected in test_cases:
            with self.subTest(input_domain=input_domain):
                result = recon.sanitize_domain(input_domain)
                self.assertEqual(result, expected)
    
    def test_protocol_removal(self):
        """Test removal of protocols from domains."""
        test_cases = [
            ("https://example.com", "example.com"),
            ("http://example.com", "example.com"),
            ("ftp://files.example.com", "files.example.com"),
        ]
        
        for input_domain, expected in test_cases:
            with self.subTest(input_domain=input_domain):
                result = recon.sanitize_domain(input_domain)
                self.assertEqual(result, expected)
    
    def test_port_removal(self):
        """Test removal of ports from domains."""
        test_cases = [
            ("example.com:8080", "example.com"),
            ("192.168.1.1:443", "192.168.1.1"),
            ("https://example.com:8443", "example.com"),
        ]
        
        for input_domain, expected in test_cases:
            with self.subTest(input_domain=input_domain):
                result = recon.sanitize_domain(input_domain)
                self.assertEqual(result, expected)
    
    def test_invalid_domains(self):
        """Test invalid domain inputs raise ValueError."""
        invalid_domains = [
            "",
            "   ",
            "invalid",
            "just-text",
            "999.999.999.999",  # Invalid IP
        ]
        
        for invalid_domain in invalid_domains:
            with self.subTest(invalid_domain=invalid_domain):
                with self.assertRaises(ValueError):
                    recon.sanitize_domain(invalid_domain)


class TestIPResolver(unittest.TestCase):
    """Test IP address resolution functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.resolver = recon.IPResolver()
    
    def test_localhost_resolution(self):
        """Test resolution of localhost."""
        ipv4 = self.resolver.resolve_ipv4("localhost")
        self.assertEqual(ipv4, "127.0.0.1")
        
        ipv6 = self.resolver.resolve_ipv6("localhost")
        self.assertEqual(ipv6, "::1")
    
    @patch('socket.gethostbyname')
    def test_ipv4_resolution_failure(self, mock_gethostbyname):
        """Test IPv4 resolution failure handling."""
        mock_gethostbyname.side_effect = socket.gaierror("Name resolution failed")
        
        result = self.resolver.resolve_ipv4("nonexistent.invalid")
        self.assertIsNone(result)
    
    @patch('socket.getaddrinfo')
    def test_ipv6_resolution_failure(self, mock_getaddrinfo):
        """Test IPv6 resolution failure handling."""
        mock_getaddrinfo.side_effect = socket.gaierror("Name resolution failed")
        
        result = self.resolver.resolve_ipv6("nonexistent.invalid")
        self.assertIsNone(result)


class TestPortScanner(unittest.TestCase):
    """Test port scanning functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scanner = recon.PortScanner()
    
    def test_service_identification(self):
        """Test service identification by port number."""
        test_cases = [
            (22, "SSH"),
            (80, "HTTP"),
            (443, "HTTPS"),
            (3389, "RDP"),
            (9999, "unknown"),  # Unknown port
        ]
        
        for port, expected_service in test_cases:
            with self.subTest(port=port):
                service = self.scanner._identify_service(port)
                self.assertEqual(service, expected_service)
    
    @patch('socket.socket')
    def test_port_scan_open(self, mock_socket_class):
        """Test port scanning when port is open."""
        mock_socket = Mock()
        mock_socket.connect_ex.return_value = 0  # Port is open
        mock_socket_class.return_value = mock_socket
        
        result = self.scanner.scan_port("127.0.0.1", 80)
        self.assertTrue(result)
        mock_socket.connect_ex.assert_called_once_with(("127.0.0.1", 80))
        mock_socket.close.assert_called_once()
    
    @patch('socket.socket')
    def test_port_scan_closed(self, mock_socket_class):
        """Test port scanning when port is closed."""
        mock_socket = Mock()
        mock_socket.connect_ex.return_value = 1  # Port is closed
        mock_socket_class.return_value = mock_socket
        
        result = self.scanner.scan_port("127.0.0.1", 9999)
        self.assertFalse(result)


class TestHTTPHeaderAnalyzer(unittest.TestCase):
    """Test HTTP header analysis functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.session = Mock()
        self.analyzer = recon.HTTPHeaderAnalyzer(self.session)
    
    def test_technology_detection(self):
        """Test technology stack detection from headers."""
        test_headers = {
            'server': 'nginx/1.18.0',
            'x-powered-by': 'PHP/7.4.3',
            'x-generator': 'WordPress 5.8',
            'x-frame-options': 'SAMEORIGIN',
        }
        
        technologies = self.analyzer.detect_technology_stack(test_headers)
        
        expected_techs = [
            "Generator: WordPress 5.8",
            "Security: x-frame-options",
            "Server: nginx/1.18.0",
            "X-Powered-By: PHP/7.4.3"
        ]
        
        self.assertEqual(sorted(technologies), sorted(expected_techs))
    
    def test_empty_headers(self):
        """Test technology detection with empty headers."""
        technologies = self.analyzer.detect_technology_stack({})
        self.assertEqual(technologies, [])


class TestWHOISAnalyzer(unittest.TestCase):
    """Test WHOIS analysis functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = recon.WHOISAnalyzer()
    
    def test_email_extraction(self):
        """Test email address extraction from WHOIS data."""
        sample_whois = """
        Registrant Name: John Doe
        Registrant Email: admin@example.com
        Technical Contact: tech@example.com
        Privacy Email: privacy@whoisguard.com
        Admin Contact: ADMIN@EXAMPLE.COM
        """
        
        emails = self.analyzer.extract_emails(sample_whois)
        
        # Should extract emails but filter out privacy services and deduplicate
        expected_emails = ["ADMIN@EXAMPLE.COM", "admin@example.com", "tech@example.com"]
        self.assertEqual(sorted(emails), sorted(expected_emails))
    
    def test_privacy_email_filtering(self):
        """Test filtering of privacy service emails."""
        privacy_whois = """
        Admin Email: privacy@domainsbyproxy.com
        Tech Email: contact@privacyguardian.org
        """
        
        emails = self.analyzer.extract_emails(privacy_whois)
        self.assertEqual(emails, [])  # Should filter out all privacy emails
    
    @patch('subprocess.run')
    def test_whois_command_not_found(self, mock_run):
        """Test handling when whois command is not available."""
        mock_run.side_effect = FileNotFoundError("whois command not found")
        
        result = self.analyzer.perform_lookup("example.com")
        self.assertEqual(result, "WHOIS command not available")


class TestReconEngine(unittest.TestCase):
    """Test the main reconnaissance engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = recon.ReconEngine()
    
    @patch.object(recon.IPResolver, 'resolve_ipv4')
    @patch.object(recon.IPResolver, 'resolve_ipv6')
    @patch.object(recon.HTTPHeaderAnalyzer, 'fetch_headers')
    @patch.object(recon.WHOISAnalyzer, 'perform_lookup')
    def test_comprehensive_recon_structure(self, mock_whois, mock_headers, mock_ipv6, mock_ipv4):
        """Test the structure of comprehensive reconnaissance output."""
        # Mock return values
        mock_ipv4.return_value = "192.168.1.1"
        mock_ipv6.return_value = "2001:db8::1"
        mock_headers.return_value = {"server": "nginx"}
        mock_whois.return_value = "Sample WHOIS data with admin@test.com"
        
        result = self.engine.perform_comprehensive_recon("test.com", enable_port_scan=False)
        
        # Verify result structure
        required_keys = [
            "domain", "timestamp", "scan_duration", "ip_addresses",
            "http_analysis", "whois_data", "network_scan", "errors"
        ]
        
        for key in required_keys:
            with self.subTest(key=key):
                self.assertIn(key, result)
        
        # Verify nested structure
        self.assertIn("ipv4", result["ip_addresses"])
        self.assertIn("ipv6", result["ip_addresses"])
        self.assertIn("technology_stack", result["http_analysis"])
        self.assertIn("emails", result["whois_data"])
        self.assertIn("open_ports", result["network_scan"])
    
    def test_timestamp_format(self):
        """Test timestamp format is ISO8601 with Z suffix."""
        with patch.object(recon.IPResolver, 'resolve_ipv4', return_value=None):
            result = self.engine.perform_comprehensive_recon("test.com", enable_port_scan=False)
            
            timestamp = result["timestamp"]
            self.assertTrue(timestamp.endswith("Z"))
            self.assertRegex(timestamp, r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z$')


class TestCLIInterface(unittest.TestCase):
    """Test command-line interface functionality."""
    
    def test_argument_parser_creation(self):
        """Test argument parser creation and basic functionality."""
        parser = recon.create_argument_parser()
        
        # Test required domain argument
        args = parser.parse_args(["example.com"])
        self.assertEqual(args.domain, "example.com")
        
        # Test optional arguments
        args = parser.parse_args([
            "example.com",
            "--verbose",
            "--no-scan",
            "--output", "test.json",
            "--proxy", "http://127.0.0.1:8080",
            "--header", "X-Test:Value",
            "--user-agent", "Custom-Agent/1.0",
            "--timeout", "2.0",
            "--log-level", "DEBUG"
        ])
        
        self.assertTrue(args.verbose)
        self.assertTrue(args.no_scan)
        self.assertEqual(args.output, "test.json")
        self.assertEqual(args.proxy, "http://127.0.0.1:8080")
        self.assertEqual(args.header, ["X-Test:Value"])
        self.assertEqual(args.user_agent, "Custom-Agent/1.0")
        self.assertEqual(args.timeout, 2.0)
        self.assertEqual(args.log_level, "DEBUG")
    
    def test_custom_header_parsing(self):
        """Test custom header parsing functionality."""
        header_list = [
            "Authorization:Bearer token123",
            "X-Custom-Header:Custom Value",
            "Invalid-Header-Without-Colon",  # Should be ignored
        ]
        
        headers = recon.parse_custom_headers(header_list)
        
        expected = {
            "Authorization": "Bearer token123",
            "X-Custom-Header": "Custom Value"
        }
        
        self.assertEqual(headers, expected)
    
    def test_output_formatting(self):
        """Test output formatting for both verbose and compact modes."""
        sample_result = {
            "domain": "test.com",
            "timestamp": "2025-06-28T17:41:29.885124Z",
            "scan_duration": 1.23,
            "ip_addresses": {
                "ipv4": "192.168.1.1",
                "ipv6": None
            },
            "http_analysis": {
                "technology_stack": ["Server: nginx", "X-Powered-By: PHP"]
            },
            "whois_data": {
                "emails": ["admin@test.com"]
            },
            "network_scan": {
                "open_ports": {80: "HTTP", 443: "HTTPS"}
            },
            "errors": []
        }
        
        # Test verbose output (should be valid JSON)
        verbose_output = recon.format_output(sample_result, verbose=True)
        parsed_json = json.loads(verbose_output)
        self.assertEqual(parsed_json["domain"], "test.com")
        
        # Test compact output
        compact_output = recon.format_output(sample_result, verbose=False)
        self.assertIn("Domain: test.com", compact_output)
        self.assertIn("IPv4: 192.168.1.1", compact_output)
        self.assertIn("Server: nginx", compact_output)
        self.assertIn("admin@test.com", compact_output)
        self.assertIn("80 (HTTP)", compact_output)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete reconnaissance workflow."""
    
    def test_json_output_file(self):
        """Test JSON output to file functionality."""
        sample_result = {
            "domain": "test.com",
            "timestamp": "2025-06-28T17:41:29.885124Z",
            "scan_duration": 1.23,
            "ip_addresses": {"ipv4": "127.0.0.1", "ipv6": None},
            "http_analysis": {"technology_stack": []},
            "whois_data": {"emails": []},
            "network_scan": {"open_ports": {}},
            "errors": []
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        try:
            # Write JSON to file
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(sample_result, f, indent=2, ensure_ascii=False)
            
            # Read back and verify
            with open(temp_path, 'r', encoding='utf-8') as f:
                loaded_result = json.load(f)
            
            self.assertEqual(loaded_result["domain"], "test.com")
            self.assertEqual(loaded_result["ip_addresses"]["ipv4"], "127.0.0.1")
            
        finally:
            os.unlink(temp_path)


def run_tests():
    """Run the complete test suite."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestDomainSanitization,
        TestIPResolver,
        TestPortScanner,
        TestHTTPHeaderAnalyzer,
        TestWHOISAnalyzer,
        TestReconEngine,
        TestCLIInterface,
        TestIntegration,
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return exit code based on test results
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    print("=== Sentenial-X Professional Recon Test Suite ===\n")
    exit_code = run_tests()
    
    if exit_code == 0:
        print("\n✓ All tests passed successfully!")
    else:
        print("\n✗ Some tests failed. Please review the output above.")
    
    sys.exit(exit_code)