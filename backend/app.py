from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import csv

from agents.master_agent import run_system

app = Flask(__name__)
CORS(app)

# =========================
# API: Analyze Telemetry
# =========================
@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON payload"}), 400

    input_data = {
        "Air temperature [K]": float(data["airTemp"]),
        "Process temperature [K]": float(data["processTemp"]),
        "Rotational speed [rpm]": float(data["rotationalSpeed"]),
        "Torque [Nm]": float(data["torque"]),
        "Tool wear [min]": float(data["toolWear"])
    }

    result = run_system(input_data)
    return jsonify(result)

# =========================
# API: UEBA Logs
# =========================
@app.route("/api/ueba", methods=["GET"])
def api_ueba():
    logs = []
    path = os.path.join("logs", "ueba_log.csv")

    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                logs.append({
                    "timestamp": row[0],
                    "action": row[1],
                    "state": row[2]
                })

    return jsonify(logs)

# =========================
# API: RCA
# =========================
@app.route("/api/rca", methods=["GET"])
def api_rca():
    path = os.path.join("logs", "manufacturing_feedback.csv")

    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
            if rows:
                return jsonify({
                    "rootCause": rows[-1][1],
                    "capa": "Immediate inspection, tool replacement, QA feedback"
                })

    return jsonify(None)

if __name__ == "__main__":
    app.run(debug=False)
