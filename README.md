# Stroke Prediction PMA
This repo contains code for an Agile Data Science assessment. It includes a Minimum Viable Data Science Application (Version 1) built using a healthcare dataset to predict stroke risk. The project features data preprocessing, a predictive model, and a Streamlit dashboard, all developed iteratively with Agile principles.

## Setup
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

## Run Prediction App
streamlit run app_predict.py

## Run Monitoring + History
streamlit run monitor_dashboard.py

## Logging
Logs are saved to: prediction_feedback_log.csv  
Inputs are stored in JSON format in the `inputs_json` column.

