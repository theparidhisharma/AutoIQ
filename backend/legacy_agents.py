from datetime import datetime
import csv
import os

os.makedirs("logs", exist_ok=True)

def log_ueba(action):
    with open("logs/ueba_log.csv", "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(), action])

def manufacturing_feedback(root_cause):
    with open("logs/manufacturing_feedback.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(), root_cause])
