import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import uuid

from log_utils import log_event

st.set_page_config(page_title="Stroke Prediction (V1 vs V2)", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
V1_PATH = BASE_DIR / "stroke_prediction_model_v1.pkl"
V2_PATH = BASE_DIR / "stroke_prediction_model_v2.pkl"

def prediction_text(pred: int) -> str:
    return "Stroke" if int(pred) == 1 else "No Stroke"

def risk_label(prob: float) -> str:
    if prob >= 0.70:
        return "High"
    if prob >= 0.40:
        return "Medium"
    return "Low"

def probability_gauge(prob: float, label: str):
    pct = int(round(prob * 100))
    st.write(f"**Probability Gauge ({label})**")
    st.progress(pct)
    st.metric("Predicted Stroke Probability", f"{prob:.3f}")

@st.cache_resource
def load_models():
    return joblib.load(V1_PATH), joblib.load(V2_PATH)

# Session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "history" not in st.session_state:
    st.session_state.history = []

st.title("🧠 Stroke Prediction App — V1 vs V2")
st.caption(f"Session ID: `{st.session_state.session_id}`")

try:
    model_v1, model_v2 = load_models()
except Exception as e:
    st.error("❌ Failed to load model files (.pkl).")
    st.exception(e)
    st.stop()

st.subheader("Enter Patient Details")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Heart Disease", [0, 1])

with col2:
    ever_married = st.selectbox("Ever Married?", ["Yes", "No"])
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])

with col3:
    avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, value=100.0)
    bmi = st.number_input("BMI", min_value=0.0, value=25.0)
    smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

input_data = {
    "gender": gender,
    "age": age,
    "hypertension": hypertension,
    "heart_disease": heart_disease,
    "ever_married": ever_married,
    "work_type": work_type,
    "Residence_type": residence_type,
    "avg_glucose_level": avg_glucose_level,
    "bmi": bmi,
    "smoking_status": smoking_status,
}
input_df = pd.DataFrame([input_data])

st.divider()

if st.button("🔍 Predict (Run both V1 and V2)", type="primary"):
    v1_pred = int(model_v1.predict(input_df)[0])
    v1_prob = float(model_v1.predict_proba(input_df)[0][1])

    v2_pred = int(model_v2.predict(input_df)[0])
    v2_prob = float(model_v2.predict_proba(input_df)[0][1])

    record = {
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input": input_data,
        "v1_pred": v1_pred,
        "v1_prob": v1_prob,
        "v2_pred": v2_pred,
        "v2_prob": v2_prob,
        "feedback_label": "",
        "feedback_text": ""
    }
    st.session_state.history.append(record)

    # Log predictions
    log_event(st.session_state.session_id, "Model_V1", input_data, v1_pred, v1_prob)
    log_event(st.session_state.session_id, "Model_V2", input_data, v2_pred, v2_prob)

    st.success("✅ Predictions generated, stored in session state, and logged.")

if len(st.session_state.history) > 0:
    latest = st.session_state.history[-1]

    st.subheader("Latest Results")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### ✅ Model V1")
        st.write("Prediction:", prediction_text(latest["v1_pred"]))
        st.write("Risk level:", risk_label(latest["v1_prob"]))
        probability_gauge(latest["v1_prob"], "V1")

    with c2:
        st.markdown("### 🚀 Model V2")
        st.write("Prediction:", prediction_text(latest["v2_pred"]))
        st.write("Risk level:", risk_label(latest["v2_prob"]))
        probability_gauge(latest["v2_prob"], "V2")

    st.subheader("Model Comparison Table")
    compare_df = pd.DataFrame([
        {"Model": "V1", "Prediction": prediction_text(latest["v1_pred"]), "Probability": latest["v1_prob"]},
        {"Model": "V2", "Prediction": prediction_text(latest["v2_pred"]), "Probability": latest["v2_prob"]},
    ])
    st.dataframe(compare_df, use_container_width=True)
    st.bar_chart(compare_df.set_index("Model")[["Probability"]])

    st.divider()
    st.subheader("Feedback")

    fb1, fb2 = st.columns([1, 2])
    with fb1:
        feedback_label = st.selectbox("Was the prediction correct?", ["", "Correct", "Incorrect"], index=0)
    with fb2:
        feedback_text = st.text_area("Optional notes", value="", height=80)

    if st.button("Submit Feedback"):
        st.session_state.history[-1]["feedback_label"] = feedback_label
        st.session_state.history[-1]["feedback_text"] = feedback_text

        # Log feedback for BOTH models
        log_event(st.session_state.session_id, "Model_V1",
                  latest["input"], latest["v1_pred"], latest["v1_prob"],
                  feedback_label=feedback_label, feedback_text=feedback_text)

        log_event(st.session_state.session_id, "Model_V2",
                  latest["input"], latest["v2_pred"], latest["v2_prob"],
                  feedback_label=feedback_label, feedback_text=feedback_text)

        st.success("✅ Feedback logged for both V1 and V2.")
else:
    st.info("Make a prediction to see results, charts, and feedback.")
