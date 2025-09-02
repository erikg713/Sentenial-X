from flask import Flask, render_template, jsonify
from api.threat_api import get_threat_data

app = Flask(__name__, template_folder="templates")

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/threats')
def get_threats():
    threats = get_threat_data()
    return jsonify(threats)

if __name__ == '__main__':
    app.run(debug=True)
