from flask import Blueprint, request, jsonify
from utils.enrich import enrich_indicator
from utils.risk import calculate_risk
from utils.siem import push_to_siem

intel_bp = Blueprint('intel', __name__)

# In-memory intel store (swap for DB in production)
intel_store = []

@intel_bp.route('/', methods=['POST'])
def submit_intel():
    """
    Accepts a new threat intelligence record, enriches it, calculates risk,
    and pushes to SIEM if applicable.
    """
    data = request.get_json()
    if not data or 'indicator' not in data:
        return jsonify({'message': 'Indicator is required'}), 400

    # Step 1: Enrich indicator with threat intel context
    enriched = enrich_indicator(data['indicator'])

    # Step 2: Calculate risk score based on intel attributes
    risk_score = calculate_risk(enriched)

    # Step 3: Save locally (replace with DB insert)
    record = {**data, **enriched, 'risk_score': risk_score}
    intel_store.append(record)

    # Step 4: Optionally forward to SIEM
    siem_result = push_to_siem(record, risk_score)

    return jsonify({
        'message': 'Intel processed successfully',
        'risk_score': risk_score,
        'siem_id': siem_result.get('id'),
        'enriched_data': enriched
    }), 201

@intel_bp.route('/', methods=['GET'])
def list_intel():
    """
    Lists all threat intel records currently in store.
    """
    return jsonify(intel_store), 200

@intel_bp.route('/<indicator>', methods=['GET'])
def get_intel(indicator):
    """
    Retrieves a specific intel record by indicator.
    """
    for record in intel_store:
        if record.get('indicator') == indicator:
            return jsonify(record), 200
    return jsonify({'message': 'Indicator not found'}), 404
