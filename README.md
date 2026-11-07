# Telecom Customer Churn Prediction

Predicts which telecom customers are likely to cancel their service, so a retention team can reach out before they leave. Includes an interactive Streamlit app for scoring individual customers.

**Dataset:** IBM Telco Customer Churn (7,043 customers, 20 features, 26.5% churn rate)
**Tech stack:** Python, pandas, scikit-learn, matplotlib, seaborn, Streamlit

## Project structure
```
churn_project/
├── churn_prediction.py   # training pipeline: cleaning, EDA, modeling, evaluation
├── app.py                # Streamlit app (user input -> churn prediction)
├── churn_utils.py        # prediction, explanation and retention-action logic used by the app
├── churn_model.joblib    # saved trained model
├── results.json          # metrics and the tuned decision threshold
├── figures/              # EDA and evaluation charts
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── requirements.txt
```

## Approach
1. **Data cleaning:** fixed `TotalCharges` stored as text (11 blank values belong to new customers with tenure 0); encoded the target.
2. **EDA:** analysed churn against tenure, contract type, internet service, payment method and support add-ons.
3. **Feature engineering:** `NumServices`, `AvgMonthlySpend`, `TenureGroup`.
4. **Modeling:** stratified 80/20 split; compared Logistic Regression, Random Forest and Gradient Boosting using 5-fold cross-validation; used class weights for the 73/27 imbalance; tuned hyperparameters with GridSearchCV.
5. **Evaluation:** ROC-AUC, precision, recall, F1, confusion matrix, permutation feature importance.
6. **Threshold tuning:** lowered the decision threshold from 0.50 to 0.29 to prioritise recall, since missing a churner costs more than a false alarm.
7. **Business view:** Low/Medium/High risk segments and top-20% lift.

## Results (held-out test set)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (tuned) | 0.733 | 0.498 | 0.791 | 0.612 | 0.842 |
| Random Forest | 0.764 | 0.539 | 0.783 | 0.638 | 0.840 |
| **Gradient Boosting (tuned)** | 0.801 | 0.668 | 0.500 | 0.572 | **0.845** |

At the tuned threshold (0.29), the selected Gradient Boosting model reaches **78% recall, 54% precision and F1 0.64**.
Customers in the High-risk segment churned at **67%** versus **7.5%** in the Low-risk segment, and the top 20% highest-risk customers contain **about 50% of all churners**.

## Key findings
- **Contract type and tenure** are the strongest churn drivers. Month-to-month customers churn at 42.7%, versus 2.8% on two-year contracts; new customers churn far more.
- **Fiber optic** users (41.9%), **electronic check** payers (45.3%) and customers **without tech support** (41.6%) churn most.

## Streamlit app
Enter a customer's details to get:
- churn probability and a Low / Medium / High risk level
- the top factors driving that customer's risk (what-if attribution against an average customer)
- rule-based retention suggestions

Run locally:
```
pip install -r requirements.txt
streamlit run app.py
```

## Retrain the model
```
python churn_prediction.py WA_Fn-UseC_-Telco-Customer-Churn.csv
```
Outputs `figures/`, `results.json` and `churn_model.joblib`.

## Limitations
- The three models score very similarly (AUC about 0.84), so the main value is the pipeline and the insights, not a large accuracy gain.
- The dataset is a single snapshot of one company, so the model may not transfer to other telecom data without retraining.
- Predictions are probability estimates for decision support, not certainties.
