# monitor_dashboard.py
# Combined: Monitoring Dashboard + History Session
# Reads prediction_feedback_log.csv created by log_utils.py (JSON-safe logging)
# Requires: streamlit, pandas, numpy

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path

# ----------------------------
# Config / Paths
# ----------------------------
st.set_page_config(page_title="Monitoring + History", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "prediction_feedback_log.csv"
DATA_PATH = BASE_DIR / "stroke_data.csv"

# ----------------------------
# Loaders
# ----------------------------
def load_baseline_data() -> pd.DataFrame:
    """Training baseline dataset used to compare input drift."""
    df = pd.read_csv(DATA_PATH)
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    if "stroke" in df.columns:
        df = df.drop(columns=["stroke"])
    return df

def load_logs(log_path: Path):
    """
    Load logs written with csv.QUOTE_ALL and a JSON column called 'inputs_json'.
    Returns:
      logs_df: the raw log rows
      inputs_df: normalized patient inputs extracted from inputs_json
    """
    if not log_path.exists():
        return pd.DataFrame(), pd.DataFrame()

    # quoting=1 corresponds to csv.QUOTE_ALL
    logs_df = pd.read_csv(log_path, quoting=1)
    logs_df["timestamp"] = pd.to_datetime(logs_df.get("timestamp"), errors="coerce")

    # Parse JSON inputs into tabular columns
    if "inputs_json" not in logs_df.columns:
        # If user still has old format, show a helpful error
        st.error("❌ Your log file does not contain the 'inputs_json' column.")
        st.info(
            "Fix: Update log_utils.py to write 'inputs_json' and regenerate prediction_feedback_log.csv.\n"
            "Then re-run the prediction app to create new logs."
        )
        return logs_df, pd.DataFrame()

    inputs_series = logs_df["inputs_json"].apply(
        lambda s: json.loads(s) if isinstance(s, str) and s.strip() else {}
    )
    inputs_df = pd.json_normalize(inputs_series)

    return logs_df, inputs_df

# ----------------------------
# Drift Metrics
# ----------------------------
def psi_numeric(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index (PSI) for numeric features."""
    expected = pd.to_numeric(pd.Series(expected), errors="coerce").dropna().to_numpy()
    actual = pd.to_numeric(pd.Series(actual), errors="coerce").dropna().to_numpy()

    if expected.size < 10 or actual.size < 10:
        return np.nan

    # Use expected quantiles to define bins
    q = np.quantile(expected, np.linspace(0, 1, bins + 1))
    q[0], q[-1] = -np.inf, np.inf

    e_counts, _ = np.histogram(expected, bins=q)
    a_counts, _ = np.histogram(actual, bins=q)

    e = e_counts / max(e_counts.sum(), 1)
    a = a_counts / max(a_counts.sum(), 1)

    eps = 1e-6
    e = np.clip(e, eps, None)
    a = np.clip(a, eps, None)

    return float(np.sum((a - e) * np.log(a / e)))

def tvd_categorical(expected: pd.Series, actual: pd.Series) -> float:
    """Total Variation Distance for categorical drift."""
    expected = expected.astype(str).fillna("NA")
    actual = actual.astype(str).fillna("NA")

    if len(expected) < 10 or len(actual) < 10:
        return np.nan

    e = expected.value_counts(normalize=True)
    a = actual.value_counts(normalize=True)

    cats = sorted(set(e.index).union(set(a.index)))
    e = e.reindex(cats, fill_value=0.0)
    a = a.reindex(cats, fill_value=0.0)

    return float(0.5 * np.abs(e - a).sum())

def safe_date_tuple(date_input):
    """Streamlit date_input can return a single date or a tuple; normalize to tuple or None."""
    if isinstance(date_input, tuple) and len(date_input) == 2:
        return date_input
    return None

# ----------------------------
# App UI
# ----------------------------
st.title("📈 Monitoring Dashboard + 🧾 History Session")

# Basic checks
if not DATA_PATH.exists():
    st.warning("⚠ stroke_data.csv not found in the same folder as this script. Drift comparisons need it.")
if not LOG_FILE.exists():
    st.info("No log file found yet. Run the prediction app and make predictions first.")
    st.stop()

baseline_df = load_baseline_data() if DATA_PATH.exists() else pd.DataFrame()
logs_df, inputs_df = load_logs(LOG_FILE)

if logs_df.empty:
    st.info("Log file exists but contains no rows yet.")
    st.stop()

# Sidebar filters (apply to BOTH tabs)
st.sidebar.header("Filters")
all_models = sorted(logs_df["model"].dropna().unique().tolist()) if "model" in logs_df.columns else []
all_sessions = sorted(logs_df["session_id"].dropna().unique().tolist()) if "session_id" in logs_df.columns else []

sel_models = st.sidebar.multiselect("Models", all_models, default=all_models)
sel_sessions = st.sidebar.multiselect("Session IDs", all_sessions, default=all_sessions)

# Date range
min_ts = logs_df["timestamp"].min()
max_ts = logs_df["timestamp"].max()
default_dates = None
if pd.notna(min_ts) and pd.notna(max_ts):
    default_dates = (min_ts.date(), max_ts.date())

date_choice = st.sidebar.date_input("Date range", value=default_dates if default_dates else None)
date_range = safe_date_tuple(date_choice)

# Build filtered mask
filtered_logs = logs_df.copy()
if sel_models:
    filtered_logs = filtered_logs[filtered_logs["model"].isin(sel_models)]
if sel_sessions:
    filtered_logs = filtered_logs[filtered_logs["session_id"].isin(sel_sessions)]
if date_range:
    start = pd.to_datetime(date_range[0])
    end = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
    filtered_logs = filtered_logs[(filtered_logs["timestamp"] >= start) & (filtered_logs["timestamp"] < end)]

# Filter inputs_df to the same rows as filtered_logs (keep index alignment)
filtered_inputs = inputs_df.loc[filtered_logs.index] if not inputs_df.empty else pd.DataFrame()

tab_monitor, tab_history = st.tabs(["📈 Monitoring Dashboard", "🧾 History Session"])

# ============================
# TAB 1: MONITORING
# ============================
with tab_monitor:
    st.subheader("Monitoring Dashboard")

    # --- Prediction drift chart
    st.markdown("### Prediction Drift (Mean probability over time)")
    if "probability" not in filtered_logs.columns or "model" not in filtered_logs.columns:
        st.warning("Log file missing required columns: 'probability' and/or 'model'.")
    else:
        prob_pivot = (
            filtered_logs
            .pivot_table(index="timestamp", columns="model", values="probability", aggfunc="mean")
            .sort_index()
        )
        st.line_chart(prob_pivot)

    # --- Drift comparisons require baseline + inputs
    st.markdown("### Data Drift (Inputs): baseline vs logged usage")
    if baseline_df.empty:
        st.info("Baseline dataset not available (stroke_data.csv missing), so drift metrics are disabled.")
    elif filtered_inputs.empty or len(filtered_inputs) < 10:
        st.warning("Not enough logged inputs to compute drift reliably. Make more predictions (10+ recommended).")
    else:
        # Align columns
        common_cols = [c for c in filtered_inputs.columns if c in baseline_df.columns]

        # Numeric PSI
        psi_rows = []
        for c in common_cols:
            base_num = pd.to_numeric(baseline_df[c], errors="coerce")
            sess_num = pd.to_numeric(filtered_inputs[c], errors="coerce")
            if base_num.notna().sum() > 10 and sess_num.notna().sum() > 10:
                psi_rows.append({"feature": c, "PSI": psi_numeric(base_num.to_numpy(), sess_num.to_numpy(), bins=10)})

        psi_df = pd.DataFrame(psi_rows).sort_values("PSI", ascending=False)
        st.write("**Numeric drift (PSI)**: >0.2 moderate drift, >0.3 high drift")
        st.dataframe(psi_df, use_container_width=True)

        # Categorical TVD (detect cat columns as non-numeric in baseline)
        tvd_rows = []
        for c in common_cols:
            # treat baseline as categorical if mostly non-numeric
            base_num_check = pd.to_numeric(baseline_df[c], errors="coerce")
            if base_num_check.notna().mean() < 0.5:  # mostly non-numeric → categorical
                tvd_rows.append({"feature": c, "TVD": tvd_categorical(baseline_df[c], filtered_inputs[c])})

        tvd_df = pd.DataFrame(tvd_rows).sort_values("TVD", ascending=False)
        st.write("**Categorical drift (TVD)**: closer to 0 stable, closer to 1 high shift")
        st.dataframe(tvd_df, use_container_width=True)

        # Simple flags
        flags = []
        if not psi_df.empty:
            high = psi_df.loc[psi_df["PSI"] > 0.30, "feature"].tolist()
            mod = psi_df.loc[(psi_df["PSI"] > 0.20) & (psi_df["PSI"] <= 0.30), "feature"].tolist()
            if high:
                flags.append(f"⚠ High numeric drift (PSI > 0.30): {', '.join(high)}")
            if mod:
                flags.append(f"⚠ Moderate numeric drift (PSI > 0.20): {', '.join(mod)}")

        if flags:
            for f in flags:
                st.warning(f)
        else:
            st.success("No major numeric drift signals detected (based on current filtered logs).")

        st.caption("Tip: Drift metrics become more meaningful with more samples (30+ logs).")

# ============================
# TAB 2: HISTORY
# ============================
with tab_history:
    st.subheader("History Session Viewer")

    st.write("Showing filtered log rows (predictions + feedback).")
    show_cols = [c for c in filtered_logs.columns if c in [
        "timestamp", "session_id", "model", "prediction", "probability", "feedback_label", "feedback_text", "inputs_json"
    ]]

    hist_view = filtered_logs[show_cols].sort_values("timestamp", ascending=False)
    st.dataframe(hist_view, use_container_width=True)

    # Download
    csv_bytes = hist_view.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Filtered Logs CSV",
        data=csv_bytes,
        file_name="prediction_feedback_log_filtered.csv",
        mime="text/csv"
    )

    st.caption("If you want a cleaner input view, you can also download and open in Excel.")
