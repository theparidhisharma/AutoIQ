import csv
from datetime import datetime

LOG_FILE = "logs/manufacturing_feedback.csv"

def log_rca(root_cause):
    try:
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now(), root_cause])
    except Exception:
        pass
