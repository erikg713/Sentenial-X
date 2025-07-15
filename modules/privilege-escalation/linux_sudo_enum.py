# modules/privilege-escalation/linux_sudo_enum.py

"""
Linux Sudo Enumeration Privilege Escalation Module

This exploit enumerates the sudo privileges for the current user by invoking
`sudo -l`, parsing the output, and reporting any allowed commands.
"""

import subprocess
import shutil
from typing import Dict, Any, List, Optional

from modules.exploits.exploit_template import Exploit, ExploitResult
from core.logging import setup_logger

logger = setup_logger(__name__, default_level="DEBUG")


class LinuxSudoEnum(Exploit):
    name = "Linux Sudo Enumeration"
    description = (
        "Enumerate sudo privileges for the current user "
        "by parsing `sudo -l` output."
    )
    target_platforms = ["Linux"]
    severity = "Low"
    requires_auth = True

    def check(self) -> bool:
        """
        Ensure `sudo` is installed and the user can run it (even if password is required).
        """
        if shutil.which("sudo") is None:
            logger.warning("'sudo' binary not found on this system.")
            return False

        logger.debug("'sudo' binary is present.")
        return True

    def exploit(self) -> ExploitResult:
        """
        Run `sudo -l`, parse the output, and return structured results.
        """
        try:
            cmd = ["sudo", "-l"]
            logger.info("Executing: %s", " ".join(cmd))
            output = subprocess.check_output(
                cmd, stderr=subprocess.STDOUT, text=True
            )
        except subprocess.CalledProcessError as e:
            logger.error("`sudo -l` failed: %s", e, exc_info=True)
            return ExploitResult(
                success=False,
                output=f"Error running `sudo -l`:\n{e.output}"
            )
        except Exception as e:
            logger.error("Unexpected error during sudo enumeration: %s", e, exc_info=True)
            return ExploitResult(success=False, output=str(e))

        entries = self._parse_sudo_output(output.splitlines())
        if not entries:
            logger.info("No sudo privileges found in output.")
            return ExploitResult(
                success=False,
                output="No sudo privileges detected.\n" + output
            )

        # Build human-readable summary
        summary_lines = [f"Found {len(entries)} sudo privilege entry(ies):"]
        for ent in entries:
            cmds = ", ".join(ent["commands"])
            summary_lines.append(
                f"  • Host: {ent['host']}, User: {ent['user']}, Commands: {cmds}"
            )
        summary = "\n".join(summary_lines)

        return ExploitResult(
            success=True,
            output=summary,
            details={"entries": entries}
        )

    def _parse_sudo_output(self, lines: List[str]) -> List[Dict[str, Any]]:
        """
        Parse lines from `sudo -l` into a list of dicts:
          [
             { "user": "<username>", "host": "<hostname>", "commands": [cmd1, cmd2, ...] },
             ...
          ]
        """
        entries: List[Dict[str, Any]] = []
        current: Optional[Dict[str, Any]] = None

        for line in lines:
            # Header: "User alice may run the following commands on myhost:"
            if line.startswith("User ") and " may run the following commands on " in line:
                # Extract user and host
                header = line[len("User "):].rstrip(":")
                user_part, host_part = header.split(" may run the following commands on ")
                current = {
                    "user": user_part.strip(),
                    "host": host_part.strip(),
                    "commands": []
                }
                entries.append(current)
                continue

            # Indented command lines follow the header
            if current and line.startswith("    "):
                cmd = line.strip()
                current["commands"].append(cmd)

        return entries
