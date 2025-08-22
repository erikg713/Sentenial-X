from flask import Blueprint, request, jsonify
from api.utils.siem import push_to_siem
from api.utils.risk_score import calculate_risk

alerts_bp = Blueprint('alerts', __name__)

@alerts_bp.route('/alert', methods=['POST'])
def create_alert():
    data = request.json
    risk = calculate_risk(data)
    enriched = push_to_siem(data, risk)
    return jsonify({"status": "success", "risk_score": risk, "siem_id": enriched["id"]})
