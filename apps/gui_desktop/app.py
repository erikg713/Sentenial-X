app.py

from flask import Flask, request, jsonify from pi_python import PiNetwork import os from dotenv import load_dotenv

load_dotenv()

app = Flask(name) pi = PiNetwork() pi.initialize()

@app.route("/payment/create", methods=[POST]) def create_payment(): data = request.json try: payment_id = pi.create_payment(data) return jsonify({"payment_id": payment_id}), 200 except Exception as e: return jsonify({"error": str(e)}), 400

@app.route("/payment/submit", methods=["POST"]) def submit_payment(): data = request.json try: txid = pi.submit_payment(data["payment_id"], data.get("pending_payments", False)) return jsonify({"txid": txid}), 200 except Exception as e: return jsonify({"error": str(e)}), 400

@app.route("/payment/complete", methods=["POST"]) def complete_payment(): data = request.json try: result = pi.complete_payment(data["payment_id"], data["txid"]) return jsonify(result), 200 except Exception as e: return jsonify({"error": str(e)}), 400

@app.route("/payment/<payment_id>", methods=["GET"]) def get_payment(payment_id): try: payment = pi.get_payment(payment_id) return jsonify(payment), 200 except Exception as e: return jsonify({"error": str(e)}), 400

@app.route("/payment/<payment_id>/cancel", methods=["POST"]) def cancel_payment(payment_id): try: result = pi.cancel_payment(payment_id) return jsonify(result), 200 except Exception as e: return jsonify({"error": str(e)}), 400

@app.route("/payments/incomplete", methods=["GET"]) def get_incomplete(): try: payments = pi.get_incomplete_server_payments() return jsonify(payments), 200 except Exception as e: return jsonify({"error": str(e)}), 400

if name == "main": app.run(debug=True)

