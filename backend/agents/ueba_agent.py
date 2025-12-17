import csv
from datetime import datetime

LOG_FILE = "logs/ueba_log.csv"

def log_action(action, state):
    try:
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now(), action, state])
    except Exception:
        pass
