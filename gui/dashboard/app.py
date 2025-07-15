from flask import Flask, render_template, jsonify, abort
from flask_cors import CORS
import os
import json
from pathlib import Path

# Config
LOG_PATH = os.getenv("THREATS_LOG_PATH", "logs/threats.json")

# Init app
app = Flask(__name__)
CORS(app)  # Optional: allow frontend requests (e.g. Vite)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/api/threats', methods=['GET'])
def get_threats():
    try:
        log_file = Path(LOG_PATH)
        if not log_file.exists():
            abort(404, description="Threat log not found.")

        with log_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)

    except json.JSONDecodeError:
        abort(500, description="Invalid JSON in threats file.")
    except Exception as e:
        abort(500, description=f"Internal error: {e}")

# Run
if __name__ == '__main__':
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, port=int(os.getenv("PORT", 5001)))
