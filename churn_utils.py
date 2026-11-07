"""Core logic for the churn app (kept separate from Streamlit so it can be tested)."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).parent
SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
INPUT_COLS = ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", "MultipleLines",
              "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
              "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges"]
FRIENDLY = {"tenure": "Tenure", "MonthlyCharges": "Monthly charges", "Contract": "Contract type",
            "InternetService": "Internet service", "PaymentMethod": "Payment method", "TechSupport": "Tech support",
            "OnlineSecurity": "Online security", "OnlineBackup": "Online backup", "DeviceProtection": "Device protection",
            "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming movies", "PaperlessBilling": "Paperless billing",
            "MultipleLines": "Multiple lines", "PhoneService": "Phone service", "SeniorCitizen": "Senior citizen",
            "Partner": "Partner", "Dependents": "Dependents", "gender": "Gender"}


def load_assets():
    model = joblib.load(BASE / "churn_model.joblib")
    cfg = json.load(open(BASE / "results.json"))
    data = pd.read_csv(BASE / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce").fillna(0)
    return model, cfg["threshold"]["threshold"], data


def finalize(df):
    """Add the engineered columns exactly as in training."""
    df = df.copy()
    df["TotalCharges"] = df["tenure"] * df["MonthlyCharges"]
    df["NumServices"] = sum((df[c] == "Yes").astype(int) for c in SERVICES)
    df["AvgMonthlySpend"] = np.where(df.tenure > 0, df.TotalCharges / df.tenure.replace(0, np.nan), df.MonthlyCharges)
    df["TenureGroup"] = pd.cut(df.tenure, [-1, 12, 24, 48, 72], labels=["0-12", "13-24", "25-48", "49-72"]).astype(str)
    return df


def predict(model, customer: dict) -> float:
    return float(model.predict_proba(finalize(pd.DataFrame([customer])))[:, 1][0])


def explain(model, customer: dict, data: pd.DataFrame, n_num=25):
    """Local what-if attribution: how much does each input move this customer's risk
    compared with an average customer's value for that input? (+ = raises churn risk)"""
    base_p = predict(model, customer)
    out = []
    for col in INPUT_COLS:
        if col in ("tenure", "MonthlyCharges"):
            vals = np.unique(np.quantile(data[col], np.linspace(0, 1, n_num)))
        else:
            vals = data[col].values
            vals = pd.Series(vals).sample(min(300, len(vals)), random_state=0).values
        rows = pd.DataFrame([customer] * len(vals))
        rows[col] = vals
        avg_p = model.predict_proba(finalize(rows))[:, 1].mean()
        out.append((col, base_p - avg_p))
    return pd.DataFrame(out, columns=["feature", "impact"]).sort_values("impact", key=abs, ascending=False)


def describe(col, customer):
    v = customer[col]
    if col == "tenure": return f"Tenure: {v} months"
    if col == "MonthlyCharges": return f"Monthly charges: ${v:.2f}"
    if col == "SeniorCitizen": return "Senior citizen: " + ("Yes" if v else "No")
    return f"{FRIENDLY[col]}: {v}"


def retention_actions(c: dict):
    acts = []
    if c["Contract"] == "Month-to-month":
        acts.append("Offer a discount to move to a 1- or 2-year contract (month-to-month customers churn at ~43% vs ~3% on 2-year).")
    if c["PaymentMethod"] == "Electronic check":
        acts.append("Nudge toward automatic payment (bank transfer / card); electronic-check payers churn most.")
    if c["InternetService"] != "No" and c["TechSupport"] == "No":
        acts.append("Offer a free trial of Tech Support / Online Security bundles.")
    if c["InternetService"] == "Fiber optic" and c["MonthlyCharges"] > 80:
        acts.append("Review pricing / speed plan for this high-bill fiber customer and offer a loyalty rate.")
    if c["tenure"] <= 12:
        acts.append("Trigger a new-customer onboarding call and early-life check-in (first-year customers are the most at risk).")
    if not acts:
        acts.append("Keep engaged with standard loyalty communication; no urgent action needed.")
    return acts
