import pandas as pd
import streamlit as st
from churn_utils import load_assets, predict, explain, describe, retention_actions, FRIENDLY

st.set_page_config(page_title="Telecom Churn Predictor", page_icon="📉", layout="wide")


@st.cache_resource
def get_assets():
    return load_assets()


model, THRESH, data = get_assets()
HIGH = 0.60

st.title("📉 Telecom Customer Churn Predictor")
st.caption("Enter a customer's details to estimate their probability of leaving, see why, and get retention suggestions. "
           "Model: tuned Gradient Boosting trained on the IBM Telco Churn dataset (ROC-AUC ≈ 0.85).")

with st.form("customer"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Account")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        payment = st.selectbox("Payment method", ["Electronic check", "Mailed check",
                                                  "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly = st.number_input("Monthly charges ($)", 18.0, 120.0, 70.0, step=0.5)
    with c2:
        st.subheader("Services")
        phone = st.selectbox("Phone service", ["Yes", "No"])
        lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        no_net = internet == "No"
        opts = ["No internet service"] if no_net else ["No", "Yes"]
        security = st.selectbox("Online security", opts)
        backup = st.selectbox("Online backup", opts)
        protection = st.selectbox("Device protection", opts)
    with c3:
        st.subheader("More services & demographics")
        support = st.selectbox("Tech support", opts)
        tv = st.selectbox("Streaming TV", opts)
        movies = st.selectbox("Streaming movies", opts)
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
    submitted = st.form_submit_button("Predict churn risk", type="primary", use_container_width=True)

if submitted:
    # keep dependent fields consistent
    if phone == "No":
        lines = "No phone service"
    if no_net:
        security = backup = protection = support = tv = movies = "No internet service"
    cust = dict(gender=gender, SeniorCitizen=int(senior == "Yes"), Partner=partner, Dependents=dependents,
                tenure=tenure, PhoneService=phone, MultipleLines=lines, InternetService=internet,
                OnlineSecurity=security, OnlineBackup=backup, DeviceProtection=protection, TechSupport=support,
                StreamingTV=tv, StreamingMovies=movies, Contract=contract, PaperlessBilling=paperless,
                PaymentMethod=payment, MonthlyCharges=monthly)
    p = predict(model, cust)
    level = "High" if p >= HIGH else ("Medium" if p >= THRESH else "Low")
    icon = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}[level]

    st.divider()
    a, b, c = st.columns(3)
    a.metric("Churn probability", f"{p:.1%}")
    b.metric("Risk level", f"{icon} {level}")
    c.metric("Model decision", "Likely to churn" if p >= THRESH else "Likely to stay",
             help=f"Decision threshold = {THRESH:.2f}, tuned to catch ~78% of churners.")
    st.progress(min(p, 1.0))
             
    left, right = st.columns(2)
    with left:
        st.subheader("Why this prediction?")
        imp = explain(model, cust, data)
        top = imp.head(6).copy()
        top["Factor"] = [describe(f, cust) for f in top.feature]
        top["Effect on risk (pp)"] = (top.impact * 100).round(1)
        st.bar_chart(top.set_index("Factor")["Effect on risk (pp)"], horizontal=True)
        st.caption("Percentage-point change in this customer's churn probability versus an average customer's value "
                   "for that factor. Positive = pushes risk up.")
        up = top[top.impact > 0].head(3).Factor.tolist()
        if up:
            st.write("**Main risk drivers:** " + "; ".join(up))
    with right:
        st.subheader("Suggested retention actions")
        for act in (retention_actions(cust) if p >= THRESH else
                    ["Customer looks stable. Keep standard loyalty communication."]):
            st.markdown(f"- {act}")
else:
    st.info("Fill in the form and click **Predict churn risk**.")

with st.expander("About this model"):
    st.write(f"Trained on 7,043 customers (26.5% churn). Gradient Boosting with class weighting by threshold tuning "
             f"(threshold {THRESH:.2f}) gives ~78% recall and ~54% precision on a held-out test set. "
             "Predictions are probabilistic estimates for decision support, not certainties.")
