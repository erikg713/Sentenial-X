import tkinter as tk
from tkinter import ttk
import threading
import time
import random

class RealTimeThreatsPanel(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.configure(style="Dark.TFrame")
        self.threat_list = []

        self.title_label = ttk.Label(self, text="Real-Time Threat Monitor", style="Heading.TLabel")
        self.title_label.pack(pady=10)

        self.tree = ttk.Treeview(self, columns=("Time", "IP", "Type", "Severity"), show="headings", height=15)
        self.tree.heading("Time", text="Timestamp")
        self.tree.heading("IP", text="Source IP")
        self.tree.heading("Type", text="Threat Type")
        self.tree.heading("Severity", text="Severity")
        self.tree.column("Time", width=130)
        self.tree.column("IP", width=120)
        self.tree.column("Type", width=150)
        self.tree.column("Severity", width=100)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.style = ttk.Style()
        self.style.configure("Treeview", background="#1e1e1e", foreground="white", fieldbackground="#1e1e1e")
        self.style.configure("Heading.TLabel", font=("Helvetica", 14, "bold"), foreground="cyan")

        # Start background thread
        self.running = True
        threading.Thread(target=self.simulate_threat_feed, daemon=True).start()

    def add_threat(self, timestamp, ip, threat_type, severity):
        self.tree.insert("", 0, values=(timestamp, ip, threat_type, severity))

    def simulate_threat_feed(self):
        fake_threats = [
            ("SQL Injection", "High"),
            ("XSS Attack", "Medium"),
            ("Port Scan", "Low"),
            ("Brute Force Login", "High"),
            ("Malicious Payload", "Critical")
        ]
        while self.running:
            time.sleep(random.randint(1, 3))
            timestamp = time.strftime("%H:%M:%S")
            ip = f"192.168.{random.randint(0,255)}.{random.randint(1,254)}"
            threat, severity = random.choice(fake_threats)
            self.add_threat(timestamp, ip, threat, severity)

    def stop(self):
        self.running = False
