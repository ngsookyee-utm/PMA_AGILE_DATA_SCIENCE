import csv
import os
from datetime import datetime

# Log file name
LOG_FILE = "prediction_logs.csv"


def log_prediction(user_input, prediction, probability):
    """
    Logs stroke prediction events for monitoring and auditing.

    Parameters:
    - user_input (dict): Patient feature inputs
    - prediction (int): Model output (0 = No Stroke, 1 = Stroke)
    - probability (float): Confidence score
    """

    # Create timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Prepare row data
    log_row = {
        "timestamp": timestamp,
        "inputs": str(user_input),
        "prediction": prediction,
        "probability": probability
    }

    # Check if file exists (write header if not)
    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=log_row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(log_row)

    print("✅ Prediction logged successfully.")
