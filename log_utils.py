import csv
import os
from datetime import datetime

LOG_FILE = "prediction_feedback_log.csv"


def log_event(model_name, user_input, prediction, probability, feedback=None):
    """
    Logs prediction + feedback events for monitoring.

    Parameters:
    - model_name (str): v1 or v2 model name
    - user_input (dict): features entered by user
    - prediction (int): model output (0 or 1)
    - probability (float): predicted probability of stroke
    - feedback (str): optional user feedback
    """

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_row = {
        "timestamp": timestamp,
        "model": model_name,
        "inputs": str(user_input),
        "prediction": prediction,
        "probability": probability,
        "feedback": feedback
    }

    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=log_row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(log_row)

