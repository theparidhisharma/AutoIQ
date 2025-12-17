import joblib
import numpy as np
import os

MODEL_PATH = "models/failure_model.pkl"
SCALER_PATH = "models/scaler.pkl"

SAFE_LIMITS = {
    "process_temp": 310,
    "rpm": 1700,
    "torque": 55,
    "tool_wear": 200
}

def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Model or scaler not found")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def rule_based_risk(data):
    score = 0
    checks = 0

    if data["Process temperature [K]"] > SAFE_LIMITS["process_temp"]:
        score += 25
    checks += 1

    if data["Rotational speed [rpm]"] > SAFE_LIMITS["rpm"]:
        score += 25
    checks += 1

    if data["Torque [Nm]"] > SAFE_LIMITS["torque"]:
        score += 25
    checks += 1

    if data["Tool wear [min]"] > SAFE_LIMITS["tool_wear"]:
        score += 25
    checks += 1

    return min(score, 100)

def compute_risk(input_data):
    model, scaler = load_artifacts()

    features = np.array([[
        input_data["Air temperature [K]"],
        input_data["Process temperature [K]"],
        input_data["Rotational speed [rpm]"],
        input_data["Torque [Nm]"],
        input_data["Tool wear [min]"]
    ]])

    features_scaled = scaler.transform(features)
    ml_prob = model.predict_proba(features_scaled)[0][1] * 100

    rule_risk = rule_based_risk(input_data)

    final_risk = (0.65 * ml_prob) + (0.35 * rule_risk)

    return round(final_risk, 2), round(ml_prob, 2), rule_risk
