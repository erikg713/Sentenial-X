# scripts/generate_cli_cheatsheet.py
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

CLI_COMMANDS = [
    ("wormgpt-detector", "Analyze adversarial AI inputs", 'sentenial_cli_full.py wormgpt-detector -p "malicious prompt"'),
    ("blindspots", "Scan for detection blind spots", "sentenial_cli_full.py blindspots"),
    ("cortex", "Run NLP-based threat analysis", 'sentenial_cli_full.py cortex -s "/var/log/syslog" -f "error"'),
    ("orchestrator", "Execute orchestrator commands", 'sentenial_cli_full.py orchestrator -a "update_policy" -p \'{"policy_id": "123"}\''),
    ("telemetry", "Stream real-time telemetry", 'sentenial_cli_full.py telemetry -s "network_monitor" -f "high_severity"'),
    ("alert", "Dispatch alerts", 'sentenial_cli_full.py alert -t "ransomware_detected" -s "high"'),
    ("simulate", "Run threat simulations", 'sentenial_cli_full.py simulate -sc "phishing_campaign"'),
]

def generate_pdf(output_path="SentenialX_CLI_CheatSheet.pdf"):
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "Sentenial-X CLI Cheat Sheet")
    c.setFont("Helvetica", 12)
    y = height - 80

    for cmd, desc, example in CLI_COMMANDS:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, f"{cmd}")
        y -= 15
        c.setFont("Helvetica", 11)
        c.drawString(60, y, f"Description: {desc}")
        y -= 15
        c.drawString(60, y, f"Example: {example}")
        y -= 25

        if y < 80:
            c.showPage()
            y = height - 50

    c.save()
    print(f"Cheat sheet PDF generated at {output_path}")

if __name__ == "__main__":
    generate_pdf()
