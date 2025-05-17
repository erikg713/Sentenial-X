import tkinter as tk
from tkinter import filedialog, messagebox
import json
import os
import numpy as np

from core.ingestor.collector import collect_sample
from core.neural_engine.profiler import get_embedding

class SentenialXGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SentenialX A.I. Threat Collector")
        self.root.geometry("700x500")
        self.root.configure(bg="#1e1e1e")  # Dark mode

        self.label = tk.Label(root, text="Choose a threat sample to analyze", font=("Helvetica", 14), fg="#ffffff", bg="#1e1e1e")
        self.label.pack(pady=20)

        self.select_button = tk.Button(root, text="Select File", command=self.load_sample, bg="#333", fg="white")
        self.select_button.pack(pady=5)

        self.save_button = tk.Button(root, text="Save Result", command=self.save_result, bg="#444", fg="white")
        self.save_button.pack(pady=5)

        self.result_frame = tk.Frame(root, bg="#1e1e1e")
        self.result_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        self.scrollbar = tk.Scrollbar(self.result_frame)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.result_text = tk.Text(self.result_frame, wrap=tk.WORD, yscrollcommand=self.scrollbar.set,
                                   height=20, width=80, bg="#2e2e2e", fg="#dcdcdc", insertbackground="white")
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar.config(command=self.result_text.yview)

        self.last_result = None  # For saving

    def load_sample(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if not file_path:
            return

        if os.path.getsize(file_path) > 5 * 1024 * 1024:  # 5MB size check
            messagebox.showwarning("File too large", "Please select a smaller file (under 5MB).")
            return

        try:
            sample_data = collect_sample(file_path)
            with open(file_path, 'r', errors='ignore') as f:
                text = f.read(1000)

            embedding = get_embedding(text)
            embedding_np = np.array(embedding)

            result = {
                "sample_info": sample_data,
                "embedding_preview": embedding[:10].tolist(),
                "embedding_info": {
                    "length": len(embedding),
                    "norm": np.linalg.norm(embedding_np).item()
                }
            }

            self.last_result = result
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, json.dumps(result, indent=2))

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def save_result(self):
        if not self.last_result:
            messagebox.showinfo("Nothing to save", "No analysis result available.")
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".json",
                                                 filetypes=[("JSON files", "*.json")])
        if save_path:
            try:
                with open(save_path, 'w') as f:
                    json.dump(self.last_result, f, indent=2)
                messagebox.showinfo("Saved", "Result saved successfully.")
            except Exception as e:
                messagebox.showerror("Error", str(e))
