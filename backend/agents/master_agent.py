from predict import compute_risk
from agents.ueba_agent import log_action
from agents.manufacturing_agent import log_rca

def classify_state(risk):
    if risk >= 90:
        return "EMERGENCY"
    if risk >= 75:
        return "CRITICAL"
    if risk >= 50:
        return "DEGRADED"
    return "NORMAL"

def run_system(input_data):
    try:
        final_risk, ml_risk, rule_risk = compute_risk(input_data)
        state = classify_state(final_risk)

        log_action("Risk Computed", state)

        if state == "EMERGENCY":
            log_rca("High thermal stress + excessive tool wear")

        return {
            "final_risk": final_risk,
            "ml_risk": ml_risk,
            "rule_risk": rule_risk,
            "state": state,
            "safe_mode": False
        }

    except Exception as e:
        log_action("System Failure", "SAFE_MODE")
        return {
            "final_risk": None,
            "ml_risk": None,
            "rule_risk": None,
            "state": "SAFE MODE",
            "safe_mode": True,
            "error": str(e)
        }
