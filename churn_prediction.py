"""Telecom Customer Churn Prediction – end-to-end pipeline.
Usage: python churn_prediction.py path/to/WA_Fn-UseC_-Telco-Customer-Churn.csv
"""
import sys, json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay,
                             precision_recall_curve)
from sklearn.inspection import permutation_importance
warnings.filterwarnings("ignore")
SEED = 42
path = sys.argv[1] if len(sys.argv) > 1 else "WA_Fn-UseC_-Telco-Customer-Churn.csv"
FIG = "figures/"

# ---------- 1. Load & clean ----------
df = pd.read_csv(path)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.loc[df["tenure"] == 0, "TotalCharges"] = 0          # 11 new customers, blank TotalCharges
df = df.drop(columns="customerID")
df["Churn"] = (df["Churn"] == "Yes").astype(int)
print("Shape:", df.shape, "| churn rate: %.2f%%" % (df.Churn.mean() * 100))

# ---------- 2. EDA ----------
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
df.Churn.map({0: "Stayed", 1: "Churned"}).value_counts().plot.bar(ax=ax[0], color=["#4C72B0", "#C44E52"]); ax[0].set_title("Class balance")
sns.histplot(data=df, x="tenure", hue=df.Churn.map({0: "Stayed", 1: "Churned"}), bins=30, ax=ax[1], multiple="stack"); ax[1].set_title("Tenure vs churn")
sns.boxplot(data=df, x=df.Churn.map({0: "Stayed", 1: "Churned"}), y="MonthlyCharges", ax=ax[2]); ax[2].set_xlabel(""); ax[2].set_title("Monthly charges vs churn")
plt.tight_layout(); plt.savefig(FIG + "01_eda_overview.png", dpi=130); plt.close()

cats = ["Contract", "InternetService", "PaymentMethod", "TechSupport", "OnlineSecurity", "PaperlessBilling"]
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for a, c in zip(axes.ravel(), cats):
    (df.groupby(c).Churn.mean() * 100).sort_values().plot.barh(ax=a, color="#C44E52")
    a.set_title(f"Churn rate (%) by {c}"); a.set_ylabel("")
plt.tight_layout(); plt.savefig(FIG + "02_churn_by_category.png", dpi=130); plt.close()

eda_rates = {c: (df.groupby(c).Churn.mean() * 100).round(1).to_dict() for c in ["Contract", "InternetService", "PaymentMethod", "TechSupport"]}

# ---------- 3. Feature engineering ----------
svc = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
df["NumServices"] = sum((df[c] == "Yes").astype(int) for c in svc)
df["AvgMonthlySpend"] = np.where(df.tenure > 0, df.TotalCharges / df.tenure.replace(0, np.nan), df.MonthlyCharges)
df["TenureGroup"] = pd.cut(df.tenure, [-1, 12, 24, 48, 72], labels=["0-12", "13-24", "25-48", "49-72"]).astype(str)

X, y = df.drop(columns="Churn"), df["Churn"]
num = ["tenure", "MonthlyCharges", "TotalCharges", "NumServices", "AvgMonthlySpend"]
cat = [c for c in X.columns if c not in num]
pre = ColumnTransformer([("num", StandardScaler(), num),
                         ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), cat)])
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

# ---------- 4. Models (class_weight handles the 73/27 imbalance) ----------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", random_state=SEED, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
rows, fitted = [], {}
for name, m in models.items():
    pipe = Pipeline([("pre", pre), ("clf", m)])
    cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc").mean()
    pipe.fit(X_tr, y_tr); fitted[name] = pipe
    p = pipe.predict_proba(X_te)[:, 1]; pred = (p >= 0.5).astype(int)
    rows.append(dict(Model=name, CV_AUC=cv_auc, Accuracy=accuracy_score(y_te, pred), Precision=precision_score(y_te, pred),
                     Recall=recall_score(y_te, pred), F1=f1_score(y_te, pred), ROC_AUC=roc_auc_score(y_te, p)))
res = pd.DataFrame(rows).round(3); print(res.to_string(index=False))

# ---------- 5. Tune best candidate (Logistic Regression is strong on this data; tune it + GB) ----------
grid = GridSearchCV(Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))]),
                    {"clf__C": [0.01, 0.05, 0.1, 0.5, 1, 5]}, cv=cv, scoring="roc_auc").fit(X_tr, y_tr)
print("Best LR C:", grid.best_params_, "CV AUC %.3f" % grid.best_score_)
fitted["Logistic Regression (tuned)"] = grid.best_estimator_
gb_grid = GridSearchCV(Pipeline([("pre", pre), ("clf", GradientBoostingClassifier(random_state=SEED))]),
                       {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.03, 0.05, 0.1], "clf__max_depth": [2, 3]},
                       cv=cv, scoring="roc_auc", n_jobs=-1).fit(X_tr, y_tr)
print("Best GB:", gb_grid.best_params_, "CV AUC %.3f" % gb_grid.best_score_)
fitted["Gradient Boosting (tuned)"] = gb_grid.best_estimator_

final_rows = []
for name in ["Logistic Regression (tuned)", "Gradient Boosting (tuned)", "Random Forest"]:
    p = fitted[name].predict_proba(X_te)[:, 1]; pred = (p >= 0.5).astype(int)
    final_rows.append(dict(Model=name, Accuracy=accuracy_score(y_te, pred), Precision=precision_score(y_te, pred),
                           Recall=recall_score(y_te, pred), F1=f1_score(y_te, pred), ROC_AUC=roc_auc_score(y_te, p)))
final = pd.DataFrame(final_rows).round(3); print(final.to_string(index=False))
best_name = final.sort_values("ROC_AUC", ascending=False).iloc[0].Model
best = fitted[best_name]; print("Selected model:", best_name)

# ---------- 6. Evaluation plots ----------
plt.figure(figsize=(6, 5))
for name in final.Model:
    fpr, tpr, _ = roc_curve(y_te, fitted[name].predict_proba(X_te)[:, 1])
    plt.plot(fpr, tpr, label=f"{name} (AUC={final.set_index('Model').loc[name,'ROC_AUC']:.3f})")
plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC curves (test set)"); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG + "03_roc_curves.png", dpi=130); plt.close()

# Threshold tuning: pick threshold maximizing F1
p_best = best.predict_proba(X_te)[:, 1]
pr, rc, th = precision_recall_curve(y_te, p_best)
f1s = 2 * pr * rc / (pr + rc + 1e-9); t_best = float(th[np.argmax(f1s[:-1])])
pred_t = (p_best >= t_best).astype(int)
thr = dict(threshold=round(t_best, 2), precision=round(precision_score(y_te, pred_t), 3), recall=round(recall_score(y_te, pred_t), 3), f1=round(f1_score(y_te, pred_t), 3))
print("Threshold-tuned:", thr)
ConfusionMatrixDisplay(confusion_matrix(y_te, pred_t), display_labels=["Stayed", "Churned"]).plot(cmap="Blues")
plt.title(f"Confusion matrix (threshold={t_best:.2f})"); plt.tight_layout(); plt.savefig(FIG + "04_confusion_matrix.png", dpi=130); plt.close()

# ---------- 7. Feature importance (permutation, on raw features -> readable) ----------
pi = permutation_importance(best, X_te, y_te, scoring="roc_auc", n_repeats=15, random_state=SEED, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=X.columns).sort_values().tail(10)
imp.plot.barh(figsize=(7, 5), color="#4C72B0"); plt.title("Top drivers of churn (permutation importance, ΔAUC)")
plt.tight_layout(); plt.savefig(FIG + "05_feature_importance.png", dpi=130); plt.close()
print("Top features:", list(imp.index[::-1][:6]))

# ---------- 8. Business view: risk segmentation ----------
te = X_te.copy(); te["churn_prob"] = p_best; te["actual"] = y_te.values
te["risk"] = pd.qcut(te.churn_prob, [0, .5, .8, 1], labels=["Low", "Medium", "High"])
seg = te.groupby("risk").agg(customers=("actual", "size"), actual_churn_rate=("actual", "mean")).round(3); print(seg)
top20 = te.sort_values("churn_prob", ascending=False).head(int(len(te) * .2))
capture = top20.actual.sum() / te.actual.sum(); print("Top-20%% riskiest customers capture %.1f%% of churners" % (capture * 100))

json.dump(dict(results=final.to_dict("records"), selected=best_name, threshold=thr, eda=eda_rates,
               top_features=list(imp.index[::-1]), top20_capture=round(float(capture), 3)),
          open("results.json", "w"), indent=2)
import joblib; joblib.dump(best, "churn_model.joblib")
