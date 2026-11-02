import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ChurnSense", page_icon="📉", layout="centered")
BASE = Path(__file__).parent


@st.cache_resource
def load_assets():
    bundle = joblib.load(BASE / "models" / "churn_model.joblib")
    meta = json.loads((BASE / "models" / "meta.json").read_text())
    return bundle["model"], float(bundle["threshold"]), meta


model, threshold, meta = load_assets()

st.title("📉 ChurnSense")
st.caption("Predict whether a customer is likely to churn. "
           "Pipeline: preprocessing + tuned classifier + threshold chosen on out-of-fold predictions.")

with st.form("customer"):
    left, right = st.columns(2)
    values = {}
    i = 0
    for name, info in meta["num"].items():
        with (left if i % 2 == 0 else right):
            if info["is_int"]:
                values[name] = st.number_input(
                    name, min_value=int(info["min"]), max_value=int(info["max"]),
                    value=int(info["median"]), step=1)
            else:
                values[name] = st.number_input(
                    name, min_value=float(info["min"]), max_value=float(info["max"]),
                    value=float(info["median"]))
        i += 1
    for name, options in meta["cat"].items():
        with (left if i % 2 == 0 else right):
            values[name] = st.selectbox(name, options)
        i += 1
    submitted = st.form_submit_button("Predict churn")

if submitted:
    row = pd.DataFrame([values])
    row = row[list(model.feature_names_in_)]   # same column order as training
    proba = float(model.predict_proba(row)[0, 1])
    is_churn = proba >= threshold

    st.subheader("Result")
    if is_churn:
        st.error(f"Likely to CHURN  (score {proba:.1%})")
    else:
        st.success(f"Likely to STAY  (score {proba:.1%})")
    st.progress(min(max(proba, 0.0), 1.0))
    st.caption(f"Decision threshold: {threshold:.3f}. The score is a model output, "
               f"not a calibrated probability.")