"""
RiskAI — Bank Loan Default Prediction & Risk Intelligence System
Machine Learning Capstone Project

A premium, dark-fintech Streamlit application that deploys a trained
RandomForestClassifier to estimate bank loan default probability and
present the result as an interactive credit-risk intelligence dashboard.
"""

import os
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="RiskAI | Bank Loan Default Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
/* Dropdown menu */
div[role="listbox"] {
    background-color: #1E293B !important;
}

/* Dropdown options */
div[role="option"] {
    background-color: #1E293B !important;
    color: white !important;
}

/* Hovered option */
div[role="option"]:hover {
    background-color: #334155 !important;
    color: white !important;
}

/* Selected option */
div[role="option"][aria-selected="true"] {
    background-color: #0F766E !important;
    color: white !important;
}

/* Selectbox text */
div[data-baseweb="select"] {
    background-color: #1E293B !important;
    color: white !important;
}

/* Selectbox selected value */
div[data-baseweb="select"] * {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "loan_default_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
DATA_PATH = os.path.join(BASE_DIR, "loan_portfolio.csv")

# Exact column order the model / scaler were fitted on
FEATURE_ORDER = [
    "Branch", "City", "State", "Gender", "Age", "Occupation", "Annual_Income",
    "Credit_Score", "Existing_Loans", "Loan_Type", "Loan_Amount", "Interest_Rate",
    "Loan_Term_Months", "EMI", "Loan_Status", "Disbursed_Amount", "Days_Past_Due",
    "Recovery_Amount", "Relationship_Years", "Account_Type", "Digital_Banking",
    "Application_Year", "Application_Month", "Approval_Year", "Approval_Month",
]

CAT_COLS = ["Branch", "City", "State", "Gender", "Occupation", "Loan_Type",
            "Loan_Status", "Account_Type", "Digital_Banking"]


# ----------------------------------------------------------------------------
# ARTIFACT / DATA LOADING
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, scaler, encoder


@st.cache_data(show_spinner=False)
def load_portfolio():
    df = pd.read_csv(DATA_PATH)
    return df


@st.cache_data(show_spinner=False)
def build_category_maps(df):
    """Reconstruct the alphabetical LabelEncoder-style mapping used at
    training time for every categorical column (sorted unique -> 0..n-1),
    exactly matching scikit-learn's LabelEncoder convention."""
    maps = {}
    for col in CAT_COLS:
        cats = sorted(df[col].dropna().unique().tolist())
        maps[col] = {v: i for i, v in enumerate(cats)}
    return maps


def calc_emi(principal, annual_rate, term_months):
    if term_months <= 0:
        return 0.0
    r = (annual_rate / 12) / 100
    if r == 0:
        return principal / term_months
    emi = principal * r * (1 + r) ** term_months / ((1 + r) ** term_months - 1)
    return round(emi, 2)


def build_feature_row(inputs, maps):
    """Assemble a single-row DataFrame in the exact column order the
    scaler/model expect, from the values collected in the form."""
    now = datetime.now()
    row = {
        "Branch": maps["Branch"][inputs["branch"]],
        "City": maps["City"][inputs["city"]],
        "State": maps["State"][inputs["state"]],
        "Gender": maps["Gender"][inputs["gender"]],
        "Age": inputs["age"],
        "Occupation": maps["Occupation"][inputs["occupation"]],
        "Annual_Income": inputs["annual_income"],
        "Credit_Score": inputs["credit_score"],
        "Existing_Loans": inputs["existing_loans"],
        "Loan_Type": maps["Loan_Type"][inputs["loan_type"]],
        "Loan_Amount": inputs["loan_amount"],
        "Interest_Rate": inputs["interest_rate"],
        "Loan_Term_Months": inputs["loan_term"],
        "EMI": calc_emi(inputs["loan_amount"], inputs["interest_rate"], inputs["loan_term"]),
        "Loan_Status": maps["Loan_Status"]["Pending"],
        "Disbursed_Amount": 0,
        "Days_Past_Due": 0,
        "Recovery_Amount": 0,
        "Relationship_Years": inputs["relationship_years"],
        "Account_Type": maps["Account_Type"][inputs["account_type"]],
        "Digital_Banking": maps["Digital_Banking"][inputs["digital_banking"]],
        "Application_Year": now.year,
        "Application_Month": now.month,
        "Approval_Year": 0,
        "Approval_Month": 0,
    }
    return pd.DataFrame([row])[FEATURE_ORDER]


@st.cache_data(show_spinner=False)
def score_portfolio_reference(_model, _scaler, df, _maps):
    """Score the historical portfolio once so we have a real, data-driven
    distribution of predicted default probabilities to calibrate the
    LOW / MODERATE / HIGH risk bands and the gauge scale against — instead
    of assuming an arbitrary 0-100% spread."""
    work = df.copy()
    for col in CAT_COLS:
        work[col] = work[col].map(_maps[col])

    app_dt = pd.to_datetime(work["Application_Date"], errors="coerce")
    appr_dt = pd.to_datetime(work["Approval_Date"], errors="coerce", dayfirst=True)
    work["Application_Year"] = app_dt.dt.year
    work["Application_Month"] = app_dt.dt.month
    work["Approval_Year"] = appr_dt.dt.year.fillna(0)
    work["Approval_Month"] = appr_dt.dt.month.fillna(0)

    X = work[FEATURE_ORDER]
    Xs = _scaler.transform(X)
    proba = _model.predict_proba(Xs)[:, 1]
    return proba


@st.cache_data(show_spinner=False)
def compute_model_metrics(_model, _scaler, df, _maps):
    """Evaluate the shipped model on a stratified 80/20 hold-out split of
    the portfolio so the Model Performance KPIs reflect the real model,
    never hard-coded numbers."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, precision_score,
                                  recall_score, f1_score, roc_auc_score)

    work = df.copy()
    for col in CAT_COLS:
        work[col] = work[col].map(_maps[col])
    app_dt = pd.to_datetime(work["Application_Date"], errors="coerce")
    appr_dt = pd.to_datetime(work["Approval_Date"], errors="coerce", dayfirst=True)
    work["Application_Year"] = app_dt.dt.year
    work["Application_Month"] = app_dt.dt.month
    work["Approval_Year"] = appr_dt.dt.year.fillna(0)
    work["Approval_Month"] = appr_dt.dt.month.fillna(0)

    X = work[FEATURE_ORDER]
    y = df["Default_Flag"].map({"No": 0, "Yes": 1})

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    Xte_s = _scaler.transform(Xte)
    preds = _model.predict(Xte_s)
    proba = _model.predict_proba(Xte_s)[:, 1]

    return {
        "accuracy": accuracy_score(yte, preds),
        "precision": precision_score(yte, preds, zero_division=0),
        "recall": recall_score(yte, preds, zero_division=0),
        "f1": f1_score(yte, preds, zero_division=0),
        "roc_auc": roc_auc_score(yte, proba),
        "default_rate": float(y.mean()),
    }


def risk_band(prob, ref_probs):
    """Classify a probability into LOW / MODERATE / HIGH using percentile
    cut-offs derived from the real portfolio distribution."""
    p60, p90 = np.percentile(ref_probs, [60, 90])
    if prob >= p90:
        return "HIGH"
    elif prob >= p60:
        return "MODERATE"
    return "LOW"


# ----------------------------------------------------------------------------
# GLOBAL CSS — DARK FINTECH THEME
# ----------------------------------------------------------------------------
def inject_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

        html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

        .stApp {
            background: radial-gradient(circle at 15% 0%, rgba(59,130,246,0.10), transparent 45%),
                        radial-gradient(circle at 85% 15%, rgba(34,211,238,0.08), transparent 40%),
                        radial-gradient(circle at 50% 100%, rgba(59,130,246,0.06), transparent 50%),
                        linear-gradient(180deg, #070B14 0%, #0A0F1D 50%, #070B14 100%);
            background-attachment: fixed;
        }

        /* subtle grid overlay */
        .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            background-image:
                linear-gradient(rgba(148,163,184,0.035) 1px, transparent 1px),
                linear-gradient(90deg, rgba(148,163,184,0.035) 1px, transparent 1px);
            background-size: 42px 42px;
            pointer-events: none;
            z-index: 0;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0B1220 0%, #0D1424 100%);
            border-right: 1px solid rgba(59,130,246,0.15);
        }

        h1, h2, h3, h4, h5, h6 { color: #F8FAFC !important; font-weight: 700; }
        p, span, label, div { color: #E2E8F0; }
        .stMarkdown p { color: #94A3B8; }

        /* ---------- Hero ---------- */
        .hero-wrap {
            background: linear-gradient(135deg, #0B1220, #111C35 60%, #0F1B33);
            border: 1px solid rgba(59,130,246,0.25);
            border-radius: 22px;
            padding: 3rem 2.5rem;
            margin-bottom: 1.75rem;
            position: relative;
            overflow: hidden;
            box-shadow: 0 0 60px rgba(59,130,246,0.08), inset 0 1px 0 rgba(255,255,255,0.04);
        }
        .hero-wrap::after {
            content: "";
            position: absolute; top: -40%; right: -10%;
            width: 480px; height: 480px; border-radius: 50%;
            background: radial-gradient(circle, rgba(34,211,238,0.18), transparent 70%);
            filter: blur(10px);
        }
        .hero-badge {
            display: inline-flex; align-items: center; gap: 8px;
            background: rgba(59,130,246,0.12); border: 1px solid rgba(59,130,246,0.35);
            color: #93C5FD; padding: 6px 14px; border-radius: 999px;
            font-size: 0.78rem; font-weight: 600; letter-spacing: 0.03em;
            margin-bottom: 1.1rem;
        }
        .hero-title {
            font-size: 3rem; font-weight: 800; line-height: 1.12; margin: 0;
            background: linear-gradient(90deg, #F8FAFC 30%, #93C5FD 75%, #22D3EE 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .hero-sub { font-size: 1.35rem; font-weight: 600; color: #93C5FD; margin-top: 0.4rem; }
        .hero-desc { color: #94A3B8; font-size: 1.02rem; max-width: 640px; margin-top: 0.9rem; line-height: 1.6; }

        /* ---------- Generic glass card ---------- */
        .glass-card {
            background: linear-gradient(135deg, rgba(17,26,46,0.85), rgba(22,33,58,0.65));
            border: 1px solid rgba(59,130,246,0.18);
            border-radius: 18px;
            padding: 1.6rem 1.5rem;
            backdrop-filter: blur(6px);
            transition: all 0.25s ease;
            height: 100%;
        }
        .glass-card:hover {
            border-color: rgba(59,130,246,0.45);
            box-shadow: 0 0 28px rgba(59,130,246,0.14);
            transform: translateY(-2px);
        }
        .glass-card h4 { margin: 0.35rem 0 0.25rem 0; font-size: 1.05rem; }
        .glass-card p { margin: 0; font-size: 0.87rem; color: #94A3B8; }
        .glass-icon { font-size: 1.6rem; }

        /* ---------- Section heading ---------- */
        .section-title {
            font-size: 1.7rem; font-weight: 800; color: #F8FAFC;
            margin: 2.4rem 0 0.3rem 0;
        }
        .section-sub { color: #94A3B8; margin-bottom: 1.4rem; font-size: 0.95rem; }

        /* ---------- Workflow step card ---------- */
        .step-card {
            background: linear-gradient(135deg, rgba(15,23,42,0.9), rgba(23,37,84,0.35));
            border: 1px solid rgba(59,130,246,0.18);
            border-radius: 16px; padding: 1.4rem 1.2rem; text-align: left; height: 100%;
        }
        .step-num {
            font-family: 'JetBrains Mono', monospace;
            color: #22D3EE; font-weight: 700; font-size: 0.85rem;
            letter-spacing: 0.08em;
        }
        .step-card h4 { margin: 0.5rem 0 0.35rem 0; font-size: 1.05rem; color: #F8FAFC; }
        .step-card p { color: #94A3B8; font-size: 0.85rem; margin: 0; }

        /* ---------- Form section card ---------- */
        .form-card-title {
            font-size: 1.15rem; font-weight: 700; color: #F8FAFC;
            border-bottom: 1px solid rgba(59,130,246,0.18);
            padding-bottom: 0.6rem; margin-bottom: 1rem;
        }

        /* ---------- Result banner ---------- */
        .result-low { background: linear-gradient(135deg, rgba(34,197,94,0.14), rgba(17,26,46,0.9)); border: 1px solid rgba(34,197,94,0.45); }
        .result-med { background: linear-gradient(135deg, rgba(245,158,11,0.14), rgba(17,26,46,0.9)); border: 1px solid rgba(245,158,11,0.45); }
        .result-high { background: linear-gradient(135deg, rgba(239,68,68,0.16), rgba(17,26,46,0.9)); border: 1px solid rgba(239,68,68,0.5); }
        .result-card {
            border-radius: 20px; padding: 2rem 2rem; text-align: center;
            box-shadow: 0 0 40px rgba(0,0,0,0.35);
        }
        .result-tag { font-size: 1rem; font-weight: 800; letter-spacing: 0.08em; }
        .result-headline { font-size: 1.6rem; font-weight: 700; color: #F8FAFC; margin: 0.3rem 0 1rem 0; }
        .result-prob { font-size: 3.4rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: #F8FAFC; }
        .result-caption { color: #94A3B8; font-size: 0.85rem; letter-spacing: 0.05em; text-transform: uppercase; }

        /* ---------- KPI metric card ---------- */
        .kpi-card {
            background: linear-gradient(135deg, rgba(17,26,46,0.9), rgba(22,33,58,0.6));
            border: 1px solid rgba(59,130,246,0.2); border-radius: 16px;
            padding: 1.2rem 1.3rem; text-align: center; height: 100%;
        }
        .kpi-label { color: #94A3B8; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; }
        .kpi-value { font-size: 1.7rem; font-weight: 800; color: #F8FAFC; font-family: 'JetBrains Mono', monospace; }

        /* ---------- badges ---------- */
        .badge-low { color: #22C55E; background: rgba(34,197,94,0.12); border: 1px solid rgba(34,197,94,0.4); padding: 3px 12px; border-radius: 999px; font-weight: 700; font-size: 0.78rem;}
        .badge-med { color: #F59E0B; background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.4); padding: 3px 12px; border-radius: 999px; font-weight: 700; font-size: 0.78rem;}
        .badge-high { color: #EF4444; background: rgba(239,68,68,0.12); border: 1px solid rgba(239,68,68,0.4); padding: 3px 12px; border-radius: 999px; font-weight: 700; font-size: 0.78rem;}

        /* ---------- disclaimer ---------- */
        .disclaimer {
            font-size: 0.78rem; color: #64748B; border-top: 1px solid rgba(148,163,184,0.15);
            padding-top: 0.8rem; margin-top: 2rem; line-height: 1.5;
        }

        /* ---------- footer ---------- */
        .app-footer {
            text-align: center; padding: 2rem 0 1rem 0; margin-top: 3rem;
            border-top: 1px solid rgba(148,163,184,0.12);
        }
        .app-footer .f-title { color: #F8FAFC; font-weight: 700; font-size: 1rem; }
        .app-footer .f-sub { color: #64748B; font-size: 0.8rem; margin-top: 0.2rem; }

        /* ---------- timeline (About) ---------- */
        .timeline-wrap { display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; margin: 1.2rem 0; }
        .timeline-node {
            background: rgba(17,26,46,0.9); border: 1px solid rgba(59,130,246,0.3);
            border-radius: 12px; padding: 0.55rem 1rem; color: #E2E8F0; font-size: 0.85rem; font-weight: 600;
        }
        .timeline-arrow { color: #3B82F6; font-size: 1.1rem; }

        /* buttons */
        .stButton>button {
            background: linear-gradient(135deg, #3B82F6, #22D3EE);
            color: #05070C; border: none; border-radius: 12px;
            font-weight: 700; padding: 0.7rem 1.4rem; letter-spacing: 0.02em;
            transition: all 0.2s ease;
        }
        .stButton>button:hover {
            box-shadow: 0 0 26px rgba(59,130,246,0.55);
            transform: translateY(-1px);
        }

        /* inputs */
        .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"], .stSlider {
            background-color: #0D1424 !important;
        }
        div[data-baseweb="select"] > div { background-color: #0D1424 !important; border-color: rgba(59,130,246,0.3) !important; }

        [data-testid="stMetricValue"] { color: #F8FAFC; }
        [data-testid="stMetricLabel"] { color: #94A3B8; }

        hr { border-color: rgba(148,163,184,0.12); }
    </style>
    """, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------------------
def sidebar_nav():
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center; padding: 0.6rem 0 1.2rem 0;">
                <div style="font-size:2.3rem;">🏦🤖</div>
                <div style="font-size:1.35rem; font-weight:800; color:#F8FAFC; letter-spacing:0.02em;">RiskAI</div>
                <div style="font-size:0.78rem; color:#94A3B8; margin-top:2px;">AI-Powered Credit Risk Intelligence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<hr style='margin:0.4rem 0 1rem 0;'>", unsafe_allow_html=True)

        pages = {
            "🏠 Home": "home",
            "🔍 Loan Risk Prediction": "predict",
            "🚨 Fraud & Risk Detection": "fraud",
            "ℹ️ About Project": "about",
        }

        if "page" not in st.session_state:
            st.session_state.page = "home"

        for label, key in pages.items():
            active = st.session_state.page == key
            if st.button(label, key=f"nav_{key}", use_container_width=True,
                         type="primary" if active else "secondary"):
                st.session_state.page = key
                st.rerun()

        st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="text-align:center; color:#64748B; font-size:0.76rem; line-height:1.5;">
                🎓 Machine Learning Capstone<br>Project
            </div>
            """,
            unsafe_allow_html=True,
        )
    return st.session_state.page


# ----------------------------------------------------------------------------
# HOME PAGE
# ----------------------------------------------------------------------------
def render_home():
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.markdown(
            """
            <div class="hero-wrap">
                <div class="hero-badge">⚡ AI-Powered Risk Engine · Live Model</div>
                <div class="hero-title">Predict Risk.<br>Protect Lending.</div>
                <div class="hero-sub">AI-Powered Bank Loan Default Prediction</div>
                <div class="hero-desc">Transform borrower information into actionable credit-risk
                intelligence using machine learning — evaluate default probability in seconds,
                not days.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔍  Start Risk Assessment", use_container_width=True):
                st.session_state.page = "predict"
                st.rerun()
        with c2:
            if st.button("📊  Explore Analytics", use_container_width=True):
                st.session_state.page = "about"
                st.rerun()

    with col2:
        render_hero_visual()

    # KPI cards
    st.markdown('<div class="section-title">Key Capabilities</div>', unsafe_allow_html=True)
    kpis = [
        ("🤖", "AI Model", "Machine Learning Powered"),
        ("📈", "Risk Analysis", "Probability-Based Prediction"),
        ("🏦", "Credit Intelligence", "Data-Driven Lending"),
        ("⚡", "Decision Support", "Real-Time Assessment"),
    ]
    cols = st.columns(4)
    for c, (icon, title, desc) in zip(cols, kpis):
        with c:
            st.markdown(
                f"""<div class="glass-card">
                        <div class="glass-icon">{icon}</div>
                        <h4>{title}</h4>
                        <p>{desc}</p>
                    </div>""",
                unsafe_allow_html=True,
            )

    # Workflow
    st.markdown('<div class="section-title">From Application to Risk Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">How the system works, end to end</div>', unsafe_allow_html=True)
    steps = [
        ("01", "Applicant Data", "Collect borrower and loan information."),
        ("02", "Data Processing", "Transform and prepare the input using the trained preprocessing pipeline."),
        ("03", "ML Prediction", "Machine learning model calculates default probability."),
        ("04", "Risk Decision", "Display risk category and probability."),
    ]
    cols = st.columns(4)
    for c, (num, title, desc) in zip(cols, steps):
        with c:
            st.markdown(
                f"""<div class="step-card">
                        <div class="step-num">STEP {num}</div>
                        <h4>{title}</h4>
                        <p>{desc}</p>
                    </div>""",
                unsafe_allow_html=True,
            )

    render_footer()


def render_hero_visual():
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=1, y1=1, xref="paper", yref="paper",
                  fillcolor="rgba(17,26,46,0.6)", line=dict(color="rgba(59,130,246,0.35)", width=1))
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=27,
        number={"suffix": "%", "font": {"color": "#F8FAFC", "size": 34}},
        title={"text": "Live Default Risk Sample", "font": {"color": "#94A3B8", "size": 13}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#334155", "tickfont": {"color": "#64748B"}},
            "bar": {"color": "#3B82F6"},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "rgba(34,197,94,0.35)"},
                {"range": [40, 70], "color": "rgba(245,158,11,0.35)"},
                {"range": [70, 100], "color": "rgba(239,68,68,0.35)"},
            ],
        },
        domain={"x": [0.08, 0.92], "y": [0.05, 0.65]},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=280, margin=dict(l=10, r=10, t=40, b=10),
        annotations=[
            dict(text="🧠 AI Neural Risk Engine", x=0.5, y=0.92, xref="paper", yref="paper",
                 showarrow=False, font=dict(color="#22D3EE", size=13)),
        ],
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="glass-card"><div class="glass-icon">💳</div><h4>Credit Score</h4><p>Real-time bureau signal</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="glass-card"><div class="glass-icon">🏦</div><h4>Loan Amount</h4><p>Exposure at origination</p></div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# PREDICTION PAGE
# ----------------------------------------------------------------------------
def render_predict(model, scaler, maps, df, ref_probs):
    st.markdown('<div class="section-title">Loan Risk Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Evaluate the probability of loan default using machine learning.</div>', unsafe_allow_html=True)

    branches = sorted(df["Branch"].unique())
    cities = sorted(df["City"].unique())
    states = sorted(df["State"].unique())
    occupations = sorted(df["Occupation"].unique())
    loan_types = sorted(df["Loan_Type"].unique())
    terms = sorted(df["Loan_Term_Months"].unique().tolist())

    with st.form("prediction_form"):
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="form-card-title">👤 Applicant Profile</div>', unsafe_allow_html=True)
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            age = st.slider("Age", 21, 60, 35)
        with a2:
            gender = st.selectbox("Gender", ["Male", "Female"])
        with a3:
            occupation = st.selectbox("Employment Status", occupations)
        with a4:
            relationship_years = st.slider("Years as Bank Customer", 0, 15, 5)
        b1, b2, b3 = st.columns(3)
        with b1:
            city = st.selectbox("City", cities)
        with b2:
            state = st.selectbox("State", states)
        with b3:
            branch = st.selectbox("Branch", branches)
        st.markdown('</div>', unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="form-card-title">💰 Financial Profile</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            annual_income = st.number_input("Annual Income (₹)", min_value=10000, max_value=1000000,
                                             value=100000, step=1000)
            st.caption(f"Monthly Income ≈ ₹{annual_income/12:,.0f}")
        with c2:
            credit_score = st.slider("Credit Score", 300, 900, 690)
        with c3:
            existing_loans = st.slider("Number of Existing Loans", 0, 6, 1)
        st.markdown('</div>', unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="form-card-title">🏦 Loan Details</div>', unsafe_allow_html=True)
        d1, d2, d3 = st.columns(3)
        with d1:
            loan_amount = st.number_input("Loan Amount (₹)", min_value=5000, max_value=1000000,
                                           value=150000, step=5000)
        with d2:
            interest_rate = st.slider("Interest Rate (%)", 5.0, 20.0, 12.0, step=0.1)
        with d3:
            loan_term = st.selectbox("Loan Term (months)", terms, index=0)
        e1, e2, e3 = st.columns(3)
        with e1:
            loan_type = st.selectbox("Loan Purpose / Type", loan_types)
        with e2:
            account_type = st.selectbox("Account Type", ["Savings", "Current"])
        with e3:
            digital_banking = st.selectbox("Digital Banking User", ["Yes", "No"])
        st.markdown('</div>', unsafe_allow_html=True)

        st.write("")
        submitted = st.form_submit_button("⚡ ANALYZE LOAN RISK", use_container_width=True)

    if submitted:
        inputs = dict(
            age=age, gender=gender, occupation=occupation, relationship_years=relationship_years,
            city=city, state=state, branch=branch, annual_income=annual_income,
            credit_score=credit_score, existing_loans=existing_loans, loan_amount=loan_amount,
            interest_rate=interest_rate, loan_term=loan_term, loan_type=loan_type,
            account_type=account_type, digital_banking=digital_banking,
        )
        with st.spinner("Analyzing applicant risk..."):
            X = build_feature_row(inputs, maps)
            Xs = scaler.transform(X)
            proba = float(model.predict_proba(Xs)[0, 1])
            pred = int(model.predict(Xs)[0])

        st.session_state.last_inputs = inputs
        st.session_state.last_proba = proba
        st.session_state.last_pred = pred
        st.session_state.last_band = risk_band(proba, ref_probs)

    if "last_proba" in st.session_state:
        render_result(model, ref_probs)

    st.markdown(
        """<div class="disclaimer">⚠️ This application provides a machine-learning-based risk
        estimate and should be used as a decision-support tool. It should not replace professional
        credit assessment or institutional lending policies.</div>""",
        unsafe_allow_html=True,
    )
    render_footer()


def render_result(model, ref_probs):
    proba = st.session_state.last_proba
    band = st.session_state.last_band
    pct = proba * 100

    band_meta = {
        "LOW": ("result-low", "🟢", "LOW RISK", "Loan Default: Unlikely"),
        "MODERATE": ("result-med", "🟡", "MODERATE RISK", "Loan Requires Additional Review"),
        "HIGH": ("result-high", "🔴", "HIGH RISK", "Potential Loan Default"),
    }
    css_class, icon, tag, headline = band_meta[band]

    st.write("")
    st.markdown(
        f"""
        <div class="result-card {css_class}">
            <div class="result-tag">{icon} {tag}</div>
            <div class="result-headline">{headline}</div>
            <div class="result-caption">Default Probability</div>
            <div class="result-prob">{pct:.2f}%</div>
            <div style="margin-top:0.8rem;">Risk Level: <b>{band}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    percentile = float((ref_probs < proba).mean() * 100)
    st.caption(f"This applicant's estimated default probability is higher than {percentile:.0f}% "
               f"of the historical loan portfolio.")

    st.write("")
    col1, col2 = st.columns([1.1, 1])
    with col1:
        render_gauge(proba, ref_probs)
    with col2:
        render_explanation(model)


def render_gauge(proba, ref_probs):
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 📈 Risk Gauge")
    p60, p90 = np.percentile(ref_probs, [60, 90])
    gauge_max = max(float(ref_probs.max()) * 1.15, proba * 1.15, p90 * 1.3, 0.01)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba * 100,
        number={"suffix": "%", "font": {"color": "#F8FAFC", "size": 32}},
        gauge={
            "axis": {"range": [0, gauge_max * 100], "tickcolor": "#334155",
                     "tickfont": {"color": "#64748B", "size": 10}},
            "bar": {"color": "#3B82F6", "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, p60 * 100], "color": "rgba(34,197,94,0.30)"},
                {"range": [p60 * 100, p90 * 100], "color": "rgba(245,158,11,0.30)"},
                {"range": [p90 * 100, gauge_max * 100], "color": "rgba(239,68,68,0.30)"},
            ],
            "threshold": {
                "line": {"color": "#F8FAFC", "width": 2},
                "thickness": 0.8,
                "value": proba * 100,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=260, margin=dict(l=20, r=20, t=10, b=10),
        font=dict(color="#94A3B8"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        '<div style="text-align:center; color:#94A3B8; font-size:0.78rem;">'
        'LOW ───────── MODERATE ───────── HIGH</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)


def render_explanation(model):
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 🧠 Why This Prediction?")
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:6]
    names = [FEATURE_ORDER[i].replace("_", " ") for i in order]
    vals = [importances[i] for i in order]

    fig = go.Figure(go.Bar(
        x=vals[::-1], y=names[::-1], orientation="h",
        marker=dict(color="#3B82F6", line=dict(color="#22D3EE", width=1)),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=260, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, tickfont=dict(color="#64748B", size=10)),
        yaxis=dict(showgrid=False, tickfont=dict(color="#E2E8F0", size=11)),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("Relative importance of the top model features (RandomForest feature importances).")
    st.markdown('</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# FRAUD & RISK DETECTION PAGE
# ----------------------------------------------------------------------------
def render_fraud(ref_probs):
    st.markdown('<div class="section-title">Fraud & Risk Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Identify applications requiring additional risk assessment.</div>', unsafe_allow_html=True)

    st.markdown(
        """<div class="glass-card" style="margin-bottom:1.2rem;">
            <b>ℹ️ Risk Monitoring & Suspicious Application Indicators</b><br>
            <span style="font-size:0.85rem;">This dashboard does not detect fraud directly — the
            portfolio does not include a labeled fraud dataset. It surfaces suspicious-application
            indicators derived from the loan-default risk model and the applicant's financial ratios.</span>
        </div>""",
        unsafe_allow_html=True,
    )

    if "last_inputs" not in st.session_state:
        st.info("Run a **Loan Risk Prediction** first — this dashboard reflects the most recently "
                "assessed applicant.")
        render_footer()
        return

    inputs = st.session_state.last_inputs
    proba = st.session_state.last_proba
    band = st.session_state.last_band

    income_to_loan = inputs["annual_income"] / max(inputs["loan_amount"], 1)
    emi = calc_emi(inputs["loan_amount"], inputs["interest_rate"], inputs["loan_term"])
    debt_burden_ratio = (emi * 12) / max(inputs["annual_income"], 1)

    if inputs["credit_score"] >= 750:
        credit_profile = "Strong"
    elif inputs["credit_score"] >= 650:
        credit_profile = "Moderate"
    else:
        credit_profile = "Weak"

    if debt_burden_ratio < 0.25:
        debt_level = "LOW"
    elif debt_burden_ratio < 0.45:
        debt_level = "MODERATE"
    else:
        debt_level = "HIGH"

    badge_map = {"LOW": "badge-low", "MODERATE": "badge-med", "HIGH": "badge-high"}

    cols = st.columns(5)
    kpis = [
        ("Credit Risk", band, badge_map[band]),
        ("Debt Burden", debt_level, badge_map[debt_level]),
        ("Income-to-Loan Ratio", f"{income_to_loan:.2f}x", ""),
        ("Credit Profile", credit_profile, ""),
        ("Overall Risk", band, badge_map[band]),
    ]
    for c, (label, val, badge_cls) in zip(cols, kpis):
        with c:
            val_html = f'<span class="{badge_cls}">{val}</span>' if badge_cls else f'<span class="kpi-value" style="font-size:1.15rem;">{val}</span>'
            st.markdown(
                f"""<div class="kpi-card">
                        <div class="kpi-label">{label}</div>
                        <div style="margin-top:0.5rem;">{val_html}</div>
                    </div>""",
                unsafe_allow_html=True,
            )

    st.write("")
    high_flags = sum([
        band == "HIGH",
        debt_level == "HIGH",
        credit_profile == "Weak",
        income_to_loan < 1.0,
    ])
    if high_flags >= 2:
        st.markdown(
            """<div class="glass-card" style="border-color:rgba(239,68,68,0.5);">
                <b>⚠️ Additional Review Recommended</b><br>
                <span style="font-size:0.85rem;">Applications displaying multiple high-risk
                indicators may require additional verification before lending approval.</span>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """<div class="glass-card" style="border-color:rgba(34,197,94,0.4);">
                <b>✅ No Elevated Indicators Detected</b><br>
                <span style="font-size:0.85rem;">This application does not currently show multiple
                high-risk indicators, based on the available data.</span>
            </div>""",
            unsafe_allow_html=True,
        )

    render_footer()


# ----------------------------------------------------------------------------
# ABOUT PAGE
# ----------------------------------------------------------------------------
def render_about(metrics):
    st.markdown('<div class="section-title">About RiskAI</div>', unsafe_allow_html=True)
    st.markdown(
        """<p style="max-width:760px; font-size:1rem; line-height:1.7;">RiskAI is an end-to-end
        machine learning application designed to estimate the probability of bank loan default and
        support data-driven credit-risk assessment.</p>""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Project Workflow</div>', unsafe_allow_html=True)
    steps = ["Data Collection", "Data Cleaning", "Exploratory Data Analysis", "Feature Engineering",
             "Model Training", "Model Evaluation", "Deployment"]
    html = '<div class="timeline-wrap">'
    for i, s in enumerate(steps):
        html += f'<div class="timeline-node">{s}</div>'
        if i != len(steps) - 1:
            html += '<span class="timeline-arrow">➜</span>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Technology Stack</div>', unsafe_allow_html=True)
    techs = ["🐍 Python", "📊 Pandas", "🔢 NumPy", "🤖 Scikit-learn", "📈 Matplotlib",
             "📊 Seaborn", "📉 Plotly", "🎨 Streamlit", "🚀 Machine Learning"]
    cols = st.columns(3)
    for i, t in enumerate(techs):
        with cols[i % 3]:
            st.markdown(f'<div class="glass-card" style="text-align:center; margin-bottom:0.8rem;"><h4>{t}</h4></div>',
                        unsafe_allow_html=True)

    st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Computed on a stratified 80/20 hold-out split of the loan portfolio.</div>', unsafe_allow_html=True)
    cols = st.columns(5)
    kpi_vals = [
        ("Accuracy", metrics["accuracy"]),
        ("Precision", metrics["precision"]),
        ("Recall", metrics["recall"]),
        ("F1 Score", metrics["f1"]),
        ("ROC-AUC", metrics["roc_auc"]),
    ]
    for c, (label, val) in zip(cols, kpi_vals):
        with c:
            st.markdown(
                f"""<div class="kpi-card">
                        <div class="kpi-label">{label}</div>
                        <div class="kpi-value">{val*100:.1f}%</div>
                    </div>""",
                unsafe_allow_html=True,
            )
    st.caption(f"⚠️ The portfolio is highly imbalanced — only {metrics['default_rate']*100:.1f}% of "
               f"historical loans defaulted — which naturally suppresses Precision/Recall at the "
               f"standard 0.5 decision threshold. ROC-AUC better reflects the model's ranking ability.")

    st.markdown('<div class="section-title">Business Impact</div>', unsafe_allow_html=True)
    impacts = [
        ("🎯", "Risk Identification", "Identify borrowers with higher estimated default probability."),
        ("🏦", "Smarter Lending", "Support data-driven credit decisions."),
        ("🛡️", "Loss Prevention", "Help financial institutions proactively manage potential credit risk."),
    ]
    cols = st.columns(3)
    for c, (icon, title, desc) in zip(cols, impacts):
        with c:
            st.markdown(
                f"""<div class="glass-card"><div class="glass-icon">{icon}</div>
                        <h4>{title}</h4><p>{desc}</p></div>""",
                unsafe_allow_html=True,
            )

    render_footer()


def render_footer():
    st.markdown(
        """
        <div class="app-footer">
            <div class="f-title">RiskAI — Bank Loan Default Prediction</div>
            <div class="f-sub">AI-Powered Credit Risk Intelligence · Machine Learning Capstone Project</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    inject_css()
    model, scaler, encoder = load_artifacts()
    df = load_portfolio()
    maps = build_category_maps(df)
    ref_probs = score_portfolio_reference(model, scaler, df, maps)

    page = sidebar_nav()

    if page == "home":
        render_home()
    elif page == "predict":
        render_predict(model, scaler, maps, df, ref_probs)
    elif page == "fraud":
        render_fraud(ref_probs)
    elif page == "about":
        metrics = compute_model_metrics(model, scaler, df, maps)
        render_about(metrics)


if __name__ == "__main__":
    main()
