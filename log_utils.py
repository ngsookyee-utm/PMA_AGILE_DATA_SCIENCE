import csv
import os
import json
from datetime import datetime

LOG_FILE = "prediction_feedback_log.csv"

FIELDNAMES = [
    "timestamp",
    "session_id",
    "model",
    "inputs_json",
    "prediction",
    "probability",
    "feedback_label",
    "feedback_text"
]

def log_event(session_id, model_name, user_input, prediction, probability,
              feedback_label="", feedback_text=""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row = {
        "timestamp": timestamp,
        "session_id": session_id,
        "model": model_name,
        "inputs_json": json.dumps(user_input, ensure_ascii=False),
        "prediction": int(prediction),
        "probability": float(probability),
        "feedback_label": feedback_label,
        "feedback_text": feedback_text
    }

    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=FIELDNAMES,
            quoting=csv.QUOTE_ALL  # ✅ prevents comma issues
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
