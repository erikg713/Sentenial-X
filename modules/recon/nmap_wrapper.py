# modules/recon/nmap_wrapper.py

import nmap
import logging

logger = logging.getLogger("sentenialx.recon.nmap")
logging.basicConfig(level=logging.INFO)

class NmapWrapper:
    def __init__(self):
        self.scanner = nmap.PortScanner()

    def scan_host(self, target_ip, arguments="-sV -T4"):
        """
        Perform a version detection and fast scan on the target.
        """
        logger.info(f"Scanning target: {target_ip} with args: {arguments}")
        try:
            self.scanner.scan(hosts=target_ip, arguments=arguments)
            return self.parse_results(target_ip)
        except Exception as e:
            logger.error(f"Scan failed: {e}")
            return None

    def parse_results(self, target_ip):
        """
        Parse the Nmap scan results into a structured dict.
        """
        if target_ip not in self.scanner.all_hosts():
            logger.warning(f"No data returned for {target_ip}")
            return {}

        host_data = {
            "ip": target_ip,
            "status": self.scanner[target_ip].state(),
            "hostname": self.scanner[target_ip].hostname(),
            "protocols": {},
        }

        for proto in self.scanner[target_ip].all_protocols():
            ports = self.scanner[target_ip][proto].keys()
            host_data["protocols"][proto] = []
            for port in sorted(ports):
                port_data = self.scanner[target_ip][proto][port]
                host_data["protocols"][proto].append({
                    "port": port,
                    "state": port_data.get("state"),
                    "name": port_data.get("name"),
                    "product": port_data.get("product", ""),
                    "version": port_data.get("version", ""),
                    "extrainfo": port_data.get("extrainfo", "")
                })
        return host_data

    def scan_multiple(self, targets):
        """
        Perform scans on multiple IPs.
        """
        results = {}
        for ip in targets:
            result = self.scan_host(ip)
            if result:
                results[ip] = result
        return results

    def get_open_ports(self, scan_result, protocol='tcp'):
        """
        Extracts only open ports from parsed scan result.
        """
        if not scan_result or protocol not in scan_result.get("protocols", {}):
            return []

        return [
            p for p in scan_result["protocols"][protocol]
            if p["state"] == "open"
        ]

# Example usage
if __name__ == "__main__":
    scanner = NmapWrapper()
    target = "192.168.1.1"
    result = scanner.scan_host(target)
    if result:
        print(f"[+] Scan result for {target}:")
        print(result)
        open_ports = scanner.get_open_ports(result)
        print(f"Open ports: {open_ports}")
    else:
        print("[-] Scan failed or no results.")
