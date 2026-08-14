# RiskAI — Bank Loan Default Prediction & Risk Intelligence System

A premium, dark-fintech Streamlit application that deploys a trained
`RandomForestClassifier` to estimate bank loan default probability and
presents it as an interactive credit-risk dashboard.

## Project Structure

```text
Bank_Loan_Default_Prediction/
│
├── app.py                     # Main Streamlit application
├── requirements.txt
├── README.md
│
├── model/
│   ├── loan_default_model.pkl # Trained RandomForestClassifier
│   ├── scaler.pkl             # StandardScaler used at training time
│   └── label_encoder.pkl      # LabelEncoder (target: Default_Flag)
│
└── data/
    └── loan_portfolio.csv     # Historical loan portfolio (reference data)
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How preprocessing was reconstructed

The shipped `scaler.pkl` / `loan_default_model.pkl` were fit on 25 features
in this exact order:

```
Branch, City, State, Gender, Age, Occupation, Annual_Income, Credit_Score,
Existing_Loans, Loan_Type, Loan_Amount, Interest_Rate, Loan_Term_Months, EMI,
Loan_Status, Disbursed_Amount, Days_Past_Due, Recovery_Amount,
Relationship_Years, Account_Type, Digital_Banking, Application_Year,
Application_Month, Approval_Year, Approval_Month
```

Categorical columns (`Branch, City, State, Gender, Occupation, Loan_Type,
Loan_Status, Account_Type, Digital_Banking`) were originally encoded with a
per-column `LabelEncoder` (alphabetical sort → 0..n-1). Only one encoder
object was saved (fit on the last column processed), so at inference time
the app rebuilds the identical alphabetical mapping directly from
`data/loan_portfolio.csv` for every categorical column — this was verified
to reproduce the exact `scaler.mean_` statistics for every feature.

`Application_Year/Month` and `Approval_Year/Month` are derived from the
`Application_Date` / `Approval_Date` columns; missing approval dates (loan
still pending/rejected) are filled with `0`, matching the fitted scaler
statistics.

For a brand-new applicant (not yet disbursed), `Loan_Status="Pending"`,
`Disbursed_Amount=0`, `Days_Past_Due=0`, and `Recovery_Amount=0` are used,
since these are unknown before a loan is issued and repaid.

**Note:** `Days_Past_Due` and `Recovery_Amount` dominate the trained model's
feature importance (~94% combined). Because both are always `0` for a new,
undisbursed application, predicted default probabilities cluster in a low
range — this is a property of the model as trained/shipped, not of the app.
The risk gauge and LOW/MODERATE/HIGH bands are calibrated against the real
distribution of scores across the historical portfolio (percentile-based),
rather than an arbitrary fixed 0–100% scale.

## Disclaimer

This application provides a machine-learning-based risk estimate and should
be used as a decision-support tool. It should not replace professional
credit assessment or institutional lending policies.
