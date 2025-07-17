import time, os
from pathlib import Path
from apps.ransomeware_emulator.payloads import sample_encrypt_payload
from sentenialx.ai_core.datastore import DB_PATH

def stream_threats():
    seen = set()
    print("[🛰️] Watching for live threats...\n")
    while True:
        threats = get_recent_threats(20)
        for row in threats:
            if row[0] not in seen:
                print(f"[{row[1]}] {row[2]} | {row[4]} | 🔥 {row[5]:.2f}")
                seen.add(row[0])
        time.sleep(1)

def scan_file(file_path):
    print(f"[📄] Scanning file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                run_scan(line.strip())

def run_simulation():
    print("[💣] Running sample ransomware payload simulation...")
    from apps.ransomeware_emulator.sandbox import RansomwareSandbox
    sandbox = RansomwareSandbox(payload_func=sample_encrypt_payload, monitor=False)
    sandbox.setup_test_environment()
    sandbox.run_payload()
    sandbox.restore_original_files()
    sandbox.cleanup()

def defend_mode():
    print("[🛡️] Defend mode: monitoring stdin for threats...")
    try:
        while True:
            line = input(">> ")
            if line.strip():
                run_scan(line.strip())
    except KeyboardInterrupt:
        print("\n[✔️] Exiting defend mode...")

def trigger_shutdown():
    Path("secure_db/SHUTDOWN.flag").touch()
    print("[⚠️] Shutdown flag created. Agents will terminate on next sync.")