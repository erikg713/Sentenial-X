from flask import Blueprint, request, jsonify
from utils.siem import push_to_siem

playbooks_bp = Blueprint('playbooks', __name__)

# In-memory store for demo (replace with DB in production)
playbooks_store = []

@playbooks_bp.route('/', methods=['POST'])
def create_playbook():
    """
    Create a new playbook with a name, description, and list of steps.
    Each step could be an action like isolate_host, block_ip, send_alert, etc.
    """
    data = request.get_json()
    required_fields = ['name', 'description', 'steps']
    if not all(field in data for field in required_fields):
        return jsonify({'message': 'Missing required fields'}), 400

    playbook = {
        'id': len(playbooks_store) + 1,
        'name': data['name'],
        'description': data['description'],
        'steps': data['steps']
    }
    playbooks_store.append(playbook)
    return jsonify({'message': 'Playbook created', 'playbook': playbook}), 201

@playbooks_bp.route('/', methods=['GET'])
def list_playbooks():
    """List all stored playbooks."""
    return jsonify(playbooks_store), 200

@playbooks_bp.route('/<int:playbook_id>', methods=['GET'])
def get_playbook(playbook_id):
    """Retrieve a specific playbook by its ID."""
    for pb in playbooks_store:
        if pb['id'] == playbook_id:
            return jsonify(pb), 200
    return jsonify({'message': 'Playbook not found'}), 404

@playbooks_bp.route('/execute/<int:playbook_id>', methods=['POST'])
def execute_playbook(playbook_id):
    """
    Simulate executing a playbook.
    In production: integrate with SOAR to run these actions for real.
    """
    for pb in playbooks_store:
        if pb['id'] == playbook_id:
            execution_log = []
            for step in pb['steps']:
                # This is where each action would be triggered
                execution_log.append(f"Executed step: {step}")
                # Optional: push execution event to SIEM
                push_to_siem({'action': step, 'playbook': pb['name']}, risk_score=0)

            return jsonify({
                'message': f'Playbook {pb["name"]} executed',
                'execution_log': execution_log
            }), 200
    return jsonify({'message': 'Playbook not found'}), 404
