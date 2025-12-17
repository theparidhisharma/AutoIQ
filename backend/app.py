from flask import Flask, request, jsonify
from flask_cors import CORS
from agents.master_agent import run_system

app = Flask(__name__)
CORS(app)

@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.json

    input_data = {
        "Air temperature [K]": data["airTemp"],
        "Process temperature [K]": data["processTemp"],
        "Rotational speed [rpm]": data["rotationalSpeed"],
        "Torque [Nm]": data["torque"],
        "Tool wear [min]": data["toolWear"]
    }

    result = run_system(input_data)

    return jsonify(result)

@app.route("/api/ueba", methods=["GET"])
def ueba():
    logs = []
    try:
        with open("logs/ueba_log.csv", newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                logs.append({
                    "timestamp": row[0],
                    "agent": "System",
                    "message": row[1],
                    "state": row[2]
                })
    except:
        pass

    return jsonify(logs)

from flask import request, jsonify
from flask_cors import CORS

# Enable CORS once
CORS(app)

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON payload received"}), 400

    input_data = {
        "Air temperature [K]": float(data["airTemp"]),
        "Process temperature [K]": float(data["processTemp"]),
        "Rotational speed [rpm]": float(data["rotationalSpeed"]),
        "Torque [Nm]": float(data["torque"]),
        "Tool wear [min]": float(data["toolWear"])
    }

    result = run_system(input_data)

    return jsonify(result)
