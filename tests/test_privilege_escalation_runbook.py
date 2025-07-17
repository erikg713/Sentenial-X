# tests/test_privilege_escalation_runbook.py

import pytest
import yaml
from pathlib import Path

# Embed the runbook YAML for testing
RUNBOOK_YAML = """
name: Privilege Escalation Campaign
description: >
  Simulates privilege escalation techniques on compromised hosts to gain
  administrative or root-level access.

stages:
  - name: Enumeration
    description: >
      Perform system enumeration to identify possible privilege escalation vectors.
    target: compromised_host
    tools:
      - name: enum_scripts.py
        type: script
        path: ../../exploits/enum_scripts.py
    parameters:
      os_type: auto
    post_actions:
      - analyze_enum_results

  - name: Linux Exploit
    description: >
      Attempt Linux-specific privilege escalation using known kernel exploits or misconfigurations.
    target: compromised_host
    prerequisites:
      - Enumeration
    tools:
      - name: linux_escalate.py
        type: script
        path: ../../exploits/linux_escalate.py
    parameters:
      exploit_timeout: 60
    post_actions:
      - verify_root_access
      - clean_temp_files

  - name: Windows Exploit
    description: >
      Attempt Windows-specific privilege escalation using local exploits or token impersonation.
    target: compromised_host
    prerequisites:
      - Enumeration
    tools:
      - name: windows_escalate.py
        type: script
        path: ../../exploits/windows_escalate.py
    parameters:
      exploit_timeout: 60
    post_actions:
      - verify_admin_access
      - clean_temp_files

verification_steps:
  - name: analyze_enum_results
    action: parse
    file: "./session_data/enum_results.json"
    criteria:
      suspicious_suid: true
      writable_configs: true

  - name: verify_root_access
    command: "id -u"
    expected_output: "0"

  - name: verify_admin_access
    command: "net session"
    expected_output_contains: "Active Sessions"

  - name: clean_temp_files
    command: "rm -rf /tmp/escalate_*"

notes:
  - Ensure exploits are tested in controlled environments to avoid crashes.
  - Log all commands and outputs for auditing.
  - Prioritize safer escalation methods before kernel exploits.
  - Consider using token impersonation on Windows before destructive techniques.
"""

@pytest.fixture
def runbook():
    return yaml.safe_load(RUNBOOK_YAML)

def test_top_level_keys(runbook):
    # Check required top-level sections
    required = {"name", "description", "stages", "verification_steps", "notes"}
    assert required.issubset(runbook.keys()), \
        f"Missing one of {required - set(runbook.keys())}"

def test_name_and_description(runbook):
    assert isinstance(runbook["name"], str) and runbook["name"].strip()
    assert isinstance(runbook["description"], str) and runbook["description"].strip()

def test_stages_structure(runbook):
    stages = runbook["stages"]
    assert isinstance(stages, list) and len(stages) >= 1

    for idx, stage in enumerate(stages):
        # Each stage must have name, description, target
        for field in ("name", "description", "target"):
            assert field in stage, f"Stage[{idx}] missing '{field}'"
            assert isinstance(stage[field], str) and stage[field].strip()

        # Tools structure
        tools = stage.get("tools", [])
        assert isinstance(tools, list) and tools, f"Stage[{idx}] must list at least one tool"
        for tool in tools:
            for tf in ("name", "type", "path"):
                assert tf in tool, f"Tool in Stage[{idx}] missing '{tf}'"
                assert isinstance(tool[tf], str) and tool[tf].strip()
            # Path should point to a .py script
            assert tool["path"].endswith(".py"), f"Tool path must end with .py: {tool['path']}"

        # Parameters should be a dict if present
        if "parameters" in stage:
            assert isinstance(stage["parameters"], dict)

        # post_actions should be a non-empty list if present
        if "post_actions" in stage:
            pa = stage["post_actions"]
            assert isinstance(pa, list) and pa, f"Stage[{idx}].post_actions must be non-empty list"

def test_verification_steps(runbook):
    steps = runbook["verification_steps"]
    assert isinstance(steps, list) and steps, "verification_steps must be a non-empty list"

    for idx, step in enumerate(steps):
        assert "name" in step and isinstance(step["name"], str)
        # either parse action with file+criteria or a command with expected output
        if step.get("action") == "parse":
            assert "file" in step and isinstance(step["file"], str)
            assert "criteria" in step and isinstance(step["criteria"], dict)
        else:
            # Must have a command
            assert "command" in step and isinstance(step["command"], str)
            # And at least one expected output field
            assert "expected_output" in step or "expected_output_contains" in step

def test_notes(runbook):
    notes = runbook["notes"]
    assert isinstance(notes, list) and notes, "notes must be a non-empty list"
    for note in notes:
        assert isinstance(note, str) and note.strip()
