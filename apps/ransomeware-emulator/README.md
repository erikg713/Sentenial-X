# Ransomware Emulator

`apps/ransomware-emulator` is a modular framework to simulate ransomware behaviors in a controlled, sandboxed environment for testing, threat emulation, and defense validation.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Features](#features)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Usage](#usage)

   * [Command-Line Interface](#command-line-interface)
   * [Python API](#python-api)
6. [Components](#components)

   * [`sandbox.py`](#sandboxpy)
   * [`emulator.py`](#emulatorpy)
   * [`behavior_patterns.py`](#behaviorpatternspy)
   * [`encryption.py`](#encryptionpy)
   * [`payloads.py`](#payloadspy)
7. [Extending the Framework](#extending-the-framework)
8. [Examples](#examples)
9. [Testing](#testing)
10. [Contributing](#contributing)
11. [License](#license)

---

## Project Structure

```
apps/ransomware-emulator/
├── sandbox.py           # Isolated test environment for payload execution
├── emulator.py          # Campaign orchestrator for running payloads
├── behavior_patterns.py # Predefined ransomware behavior building blocks
├── encryption.py        # Encryption/decryption helpers (AES, XOR, etc.)
├── payloads.py          # Registry of payload functions using behavior patterns
├── README.md            # Project documentation
└── tests/               # Unit and integration tests
```

## Features

* **Sandboxed Execution**: Emulate ransomware in a temporary, isolated directory.
* **Modular Payloads**: Define and register custom payloads via `payloads.py`.
* **Behavior Patterns**: Reusable patterns like delayed encryption, random renaming, and selective targeting.
* **Filesystem Monitoring**: Optional telemetry via sandbox monitor.
* **Extensible**: Easily add new behaviors, encryption schemes, and payloads.
* **CLI & API**: Use as a standalone CLI tool or import programmatically.

## Installation

1. Clone the repository:

   ```bash
   git clone <repository_url>
   cd apps/ransomware-emulator
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

> **Note:** Ensure `requirements.txt` includes necessary libs like `cryptography` and `watchdog` if using monitoring.

## Configuration

* **Encryption Scheme**: Configure algorithms in `encryption.py` (e.g., AES, XOR).
* **Behavior Settings**: Adjust delays, target file extensions, and randomness parameters in `behavior_patterns.py`.
* **Payload Registry**: Add or update entries in `payloads.py`.

## Usage

### Command-Line Interface

Use `emulator.py` directly:

```bash
cd apps/ransomware-emulator
python emulator.py --help
```

Example:

```bash
python emulator.py \
  --payload basic_encrypt \
  --monitor True \
  --file_count 10
```

### Python API

```python
from apps.ransomware_emulator.emulator import RansomwareEmulator

emulator = RansomwareEmulator()
emulator.run_campaign(payload_name="basic_encrypt", monitor=True, file_count=5)
print(emulator.get_results())
```

## Components

### `sandbox.py`

Provides `RansomwareSandbox` to:

* Create a temporary directory.
* Populate it with dummy files.
* Execute a payload function.
* Optionally monitor file operations.
* Cleanup after simulation.

### `emulator.py`

Offers `RansomwareEmulator` to:

* List available payloads.
* Run emulation campaigns programmatically or via CLI.
* Aggregate and return results.

### `behavior_patterns.py`

Defines functions such as:

* `rename_files_randomly`
* `delay_encryption`
* `deep_directory_traversal`
* `selective_encryption`
* `overwrite_file_contents`

Use patterns directly or compose them in payloads.

### `encryption.py`

Implement encryption and decryption helpers. Example:

```python
def encrypt_file(path: Path):
    # AES encryption logic
    pass

def decrypt_file(path: Path):
    # Decryption logic
    pass
```

### `payloads.py`

Registry mapping payload names to functions. Example:

```python
PAYLOAD_REGISTRY = {
    "basic_encrypt": basic_encrypt_payload,
    "stealth_encrypt": stealth_encrypt_payload,
}
```

## Extending the Framework

1. **Add Behavior**: Create a new function in `behavior_patterns.py`.
2. **Define Payload**: In `payloads.py`, import and combine behaviors.
3. **Test**: Write unit tests in `tests/` and ensure sandbox integration.

## Examples

Basic encrypt payload:

```python
# payloads.py
from .encryption import encrypt_file

def basic_encrypt_payload(root):
    for f in root.glob("*.txt"):
        encrypt_file(f)
        f.unlink()
```

Chaining patterns:

```python
# payloads.py
from .behavior_patterns import deep_directory_traversal, delay_encryption

def stealth_encrypt_payload(root):
    targets = deep_directory_traversal(root)
    delay_encryption(targets, delay=1.0)
```

## Testing

```bash
pytest tests/
```

Include tests for sandbox setup, payload execution, behavior functions, and cleanup.

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/new-behavior`.
3. Commit changes and push: `git push origin feature/new-behavior`.
4. Open a pull request.

Please follow the coding standards and include tests for new features.

## License

This project is licensed under the MIT License. See [LICENSE](../LICENSE) for details.
