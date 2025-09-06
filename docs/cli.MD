---

### 📂 `sentenial-x/docs/cli.md`

```markdown
# CLI Usage

Sentenial-X CLI allows direct interaction with AI analysis, simulation, and telemetry.

### Commands

| Command | Description | Example |
|---------|-------------|---------|
| cortex | Full AI threat analysis | `python sentenial_cli_full.py cortex -t "log entry"` |
| wormgpt-detector | Adversarial AI detector | `python sentenial_cli_full.py wormgpt-detector -t "malicious prompt"` |
| simulate | Trigger emulation playbooks | `python sentenial_cli_full.py simulate -t "suspicious behavior"` |

Use `--help` for detailed options:
```bash
python sentenial_cli_full.py cortex --help
