"""Streamlit demo: score a transaction with the trained model.
Run:  streamlit run app.py   (after running src/train.py)"""
import json

import joblib
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier

st.set_page_config(page_title="Fraud Detector", page_icon="💳")
st.title("💳 Credit Card Fraud Detector")


@st.cache_resource
def load():
    meta = json.load(open("models/threshold.json"))
    if meta.get("format") == "xgboost_json":
        model = XGBClassifier()
        model.load_model("models/xgb_model.json")
    else:
        model = joblib.load("models/best_model.joblib")
    sample = pd.read_csv("data/app_sample.csv")
    return model, meta, sample


model, meta, sample = load()
features = [c for c in sample.columns if c not in
            ("txn_id", "Time", "Amount", "Class", "fraud_score", "predicted_fraud")]

st.caption(f"Model: **{meta['model']}** | tuned threshold: **{meta['threshold']}**")
thr = st.slider("Decision threshold", 0.01, 0.99, float(meta["threshold"]), 0.01,
                help="Lower = catch more fraud but more false alarms")

mode = st.radio("Pick a transaction from the held-out test set",
                ["Random", "Random fraud", "Random legitimate"], horizontal=True)
if st.button("Draw transaction") or "row" not in st.session_state:
    pool = sample if mode == "Random" else sample[sample["Class"] == (1 if mode == "Random fraud" else 0)]
    st.session_state["row"] = pool.sample(1).iloc[0]

row = st.session_state["row"]
score = float(model.predict_proba(row[features].to_frame().T.astype(float))[:, 1][0])
flagged = score >= thr

c1, c2, c3 = st.columns(3)
c1.metric("Amount", f"{row['Amount']:.2f}")
c2.metric("Fraud probability", f"{score:.1%}")
c3.metric("Actual label", "Fraud" if row["Class"] == 1 else "Legitimate")

if flagged:
    st.error("🚨 FLAGGED as fraud - send for manual review")
else:
    st.success("✅ Approved")
if flagged == bool(row["Class"]):
    st.caption("The model got this one right.")
else:
    st.caption("The model got this one wrong - try changing the threshold.")
