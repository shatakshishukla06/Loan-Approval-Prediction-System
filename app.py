"""
╔══════════════════════════════════════════════════════════════╗
║         LOAN APPROVAL PREDICTION SYSTEM                      ║
║         Built with Streamlit + Scikit-learn                  ║
║         Author: B.Tech ML Project                            ║
╚══════════════════════════════════════════════════════════════╝

FILE: app.py
PURPOSE: Main Streamlit web application for loan approval prediction.
         This file handles the entire UI and connects to the trained model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIGURATION  (Must be FIRST command)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="LoanIQ · Approval System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS STYLING
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Syne:wght@600;700;800&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0f172a 40%, #0a1628 100%);
    min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #112240 100%) !important;
    border-right: 1px solid rgba(100,160,255,0.15);
}
[data-testid="stSidebar"] * { color: #c8d8f0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label { color: #7eb8f7 !important; font-weight: 500; }

/* ── Hero Header ── */
.hero-header {
    background: linear-gradient(135deg, #1a3a6b 0%, #0e2a5e 50%, #091d47 100%);
    border: 1px solid rgba(100,180,255,0.2);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.05);
}
.hero-header::before {
    content: "";
    position: absolute; top: 0; right: 0;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(59,130,246,0.12) 0%, transparent 70%);
    border-radius: 50%;
    transform: translate(30%, -30%);
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.1;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero-title span { color: #60a5fa; }
.hero-subtitle {
    font-size: 1.05rem;
    color: #94b4d4;
    margin-top: 0.6rem;
    font-weight: 400;
    letter-spacing: 0.2px;
}
.hero-badge {
    display: inline-block;
    background: rgba(96,165,250,0.15);
    border: 1px solid rgba(96,165,250,0.3);
    color: #60a5fa;
    padding: 0.3rem 0.9rem;
    border-radius: 100px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* ── Metric Cards ── */
.metric-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.02) 100%);
    border: 1px solid rgba(100,160,255,0.12);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
    backdrop-filter: blur(10px);
}
.metric-card:hover { transform: translateY(-3px); border-color: rgba(100,160,255,0.3); }
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem; font-weight: 700;
    color: #60a5fa; line-height: 1;
}
.metric-label { font-size: 0.82rem; color: #64748b; margin-top: 0.4rem; font-weight: 500; letter-spacing: 0.5px; text-transform: uppercase; }

/* ── Section Headers ── */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem; font-weight: 700;
    color: #e2e8f0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid rgba(96,165,250,0.25);
    margin-bottom: 1.2rem;
    letter-spacing: -0.2px;
}
.section-header span { color: #60a5fa; }

/* ── Info Box ── */
.info-box {
    background: rgba(30,58,100,0.35);
    border-left: 3px solid #3b82f6;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    color: #94b4d4;
    font-size: 0.9rem;
    line-height: 1.6;
}

/* ── RESULT CARDS ── */
.result-approved {
    background: linear-gradient(135deg, rgba(16,185,129,0.12) 0%, rgba(5,150,105,0.08) 100%);
    border: 2px solid rgba(16,185,129,0.4);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    animation: pulseGreen 2s ease-in-out infinite;
}
.result-rejected {
    background: linear-gradient(135deg, rgba(239,68,68,0.12) 0%, rgba(185,28,28,0.08) 100%);
    border: 2px solid rgba(239,68,68,0.4);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    animation: pulseRed 2s ease-in-out infinite;
}
@keyframes pulseGreen {
    0%,100% { box-shadow: 0 0 0 0 rgba(16,185,129,0); }
    50% { box-shadow: 0 0 0 12px rgba(16,185,129,0.08); }
}
@keyframes pulseRed {
    0%,100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
    50% { box-shadow: 0 0 0 12px rgba(239,68,68,0.08); }
}
.result-icon { font-size: 3.5rem; display: block; margin-bottom: 0.5rem; }
.result-title {
    font-family: 'Syne', sans-serif;
    font-size: 2rem; font-weight: 800;
    margin: 0.5rem 0;
}
.result-approved .result-title { color: #10b981; }
.result-rejected .result-title { color: #ef4444; }
.result-subtitle { color: #94b4d4; font-size: 0.95rem; margin-top: 0.3rem; }

/* ── Probability Bar ── */
.prob-bar-container { margin: 1rem 0; }
.prob-label { font-size: 0.85rem; color: #94b4d4; margin-bottom: 0.4rem; display: flex; justify-content: space-between; }
.prob-bar-bg { background: rgba(255,255,255,0.06); border-radius: 100px; height: 10px; overflow: hidden; }
.prob-bar-fill-green { background: linear-gradient(90deg, #10b981, #34d399); border-radius: 100px; height: 100%; transition: width 1s ease; }
.prob-bar-fill-red { background: linear-gradient(90deg, #ef4444, #f87171); border-radius: 100px; height: 100%; transition: width 1s ease; }

/* ── Feature Tag ── */
.feature-tag {
    display: inline-block;
    background: rgba(96,165,250,0.1);
    border: 1px solid rgba(96,165,250,0.2);
    color: #7eb8f7;
    padding: 0.25rem 0.7rem;
    border-radius: 6px;
    font-size: 0.78rem;
    margin: 0.2rem;
    font-weight: 500;
}

/* ── Streamlit overrides ── */
.stButton > button {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    color: white !important;
    border: none !important;
    padding: 0.75rem 2rem !important;
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    letter-spacing: 0.3px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px rgba(37,99,235,0.3) !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
    box-shadow: 0 6px 20px rgba(37,99,235,0.45) !important;
    transform: translateY(-1px) !important;
}
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input,
.stSelectbox select {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(100,160,255,0.2) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}
.stSlider [data-baseweb="slider"] { padding: 0.5rem 0; }
[data-testid="stMetric"] { background: rgba(255,255,255,0.03); border-radius: 12px; padding: 1rem; border: 1px solid rgba(100,160,255,0.1); }
[data-testid="stMetricValue"] { color: #60a5fa !important; font-family: 'Syne', sans-serif; }
[data-testid="stMetricLabel"] { color: #64748b !important; }

/* ── Tab Styles ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 12px;
    padding: 4px;
    border: 1px solid rgba(100,160,255,0.1);
}
.stTabs [data-baseweb="tab"] {
    color: #64748b !important;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: rgba(37,99,235,0.25) !important;
    color: #60a5fa !important;
}

/* ── Divider ── */
hr { border-color: rgba(100,160,255,0.1) !important; }
h1,h2,h3,h4 { color: #e2e8f0 !important; }
p, li { color: #94b4d4 !important; }
.stMarkdown p { color: #94b4d4; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(96,165,250,0.3); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HELPER: Load Trained Model & Encoders
# ─────────────────────────────────────────────
@st.cache_resource
def load_model_artifacts():
    """Load the saved model, scaler, and feature columns from disk."""
    model_path = "loan_model.pkl"
    if not os.path.exists(model_path):
        return None, None, None
    with open(model_path, "rb") as f:
        artifacts = pickle.load(f)
    return artifacts["model"], artifacts["scaler"], artifacts["feature_columns"]


# ─────────────────────────────────────────────
#  HELPER: Load Dataset for Visualizations
# ─────────────────────────────────────────────
@st.cache_data
def load_dataset():
    """Load the loan dataset for EDA visualizations."""
    paths = ["dataset/loan_data.csv", "dataset/train.csv", "dataset/loan_sanction_train.csv"]
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            return df
    return None


# ─────────────────────────────────────────────
#  HELPER: Preprocess Input for Prediction
# ─────────────────────────────────────────────
def preprocess_input(data_dict, feature_columns, scaler):
    """
    Convert raw form inputs into the feature vector the model expects.
    Applies the SAME transformations used during training.
    """
    # Encodings must match model_training.py
    gender_map      = {"Male": 1, "Female": 0}
    married_map     = {"Yes": 1, "No": 0}
    edu_map         = {"Graduate": 1, "Not Graduate": 0}
    employed_map    = {"Yes": 1, "No": 0}
    area_map        = {"Urban": 2, "Semiurban": 1, "Rural": 0}
    dependents_map  = {"0": 0, "1": 1, "2": 2, "3+": 3}

    row = {
        "Gender":             gender_map[data_dict["gender"]],
        "Married":            married_map[data_dict["married"]],
        "Dependents":         dependents_map[data_dict["dependents"]],
        "Education":          edu_map[data_dict["education"]],
        "Self_Employed":      employed_map[data_dict["self_employed"]],
        "ApplicantIncome":    data_dict["applicant_income"],
        "CoapplicantIncome":  data_dict["coapplicant_income"],
        "LoanAmount":         data_dict["loan_amount"],
        "Loan_Amount_Term":   data_dict["loan_term"],
        "Credit_History":     data_dict["credit_history"],
        "Property_Area":      area_map[data_dict["property_area"]],
        # Engineered features (must match training)
        "TotalIncome":        data_dict["applicant_income"] + data_dict["coapplicant_income"],
        "EMI":                (data_dict["loan_amount"] * 1000) / (data_dict["loan_term"] if data_dict["loan_term"] > 0 else 1),
        "IncomeAfterEMI":     (data_dict["applicant_income"] + data_dict["coapplicant_income"]) -
                              ((data_dict["loan_amount"] * 1000) / (data_dict["loan_term"] if data_dict["loan_term"] > 0 else 1)),
    }

    # Build DataFrame aligned to training columns
    df_row = pd.DataFrame([row])
    for col in feature_columns:
        if col not in df_row.columns:
            df_row[col] = 0
    df_row = df_row[feature_columns]

    # Apply same scaler from training
    df_scaled = scaler.transform(df_row)
    return df_scaled


# ─────────────────────────────────────────────
#  MATPLOTLIB DARK THEME CONFIG
# ─────────────────────────────────────────────
def set_chart_style():
    plt.rcParams.update({
        "figure.facecolor":  "#0f172a",
        "axes.facecolor":    "#0f1e35",
        "axes.edgecolor":    "#1e3a5f",
        "axes.labelcolor":   "#94b4d4",
        "xtick.color":       "#64748b",
        "ytick.color":       "#64748b",
        "text.color":        "#94b4d4",
        "grid.color":        "#1e3a5f",
        "grid.linewidth":    0.8,
        "font.family":       "DejaVu Sans",
    })


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#                   SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown("## 🏦 LoanIQ")
    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio(
        "Go to",
        ["🏠 Home & Predict", "📊 Data Analysis", "🤖 Model Info", "📋 About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem; color:#475569; line-height:1.7'>
    <b style='color:#60a5fa'>Model</b>: Logistic Regression<br>
    <b style='color:#60a5fa'>Dataset</b>: Kaggle Loan Data<br>
    <b style='color:#60a5fa'>Accuracy</b>: ~80–85%<br>
    <b style='color:#60a5fa'>Task</b>: Binary Classification
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#334155; text-align:center'>
    Built for B.Tech ML Project<br>Powered by Scikit-learn + Streamlit
    </div>
    """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#         PAGE 1 · HOME & PREDICTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if "Home" in page:

    # ── Hero ──────────────────────────────────
    st.markdown("""
    <div class="hero-header">
        <div class="hero-badge">AI-POWERED · INSTANT RESULTS</div>
        <div class="hero-title">Loan <span>Approval</span><br>Prediction System</div>
        <div class="hero-subtitle">Fill in the applicant details on the left and click <b>Predict</b> to get an instant AI-powered loan decision with confidence score.</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load model ────────────────────────────
    model, scaler, feature_columns = load_model_artifacts()
    model_loaded = model is not None

    if not model_loaded:
        st.markdown("""
        <div class="info-box">
        ⚠️ <b>Model not found.</b> Please run <code>python model_training.py</code> first to train and save the model.
        Then refresh this page.
        </div>
        """, unsafe_allow_html=True)

    # ── Top Metrics ───────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("🎯", "~83%", "Model Accuracy"),
        ("📄", "614", "Training Samples"),
        ("🔢", "11", "Input Features"),
        ("⚡", "<1s", "Prediction Time"),
    ]
    for col, (icon, val, label) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style='font-size:1.6rem'>{icon}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two-column layout: Form | Result ──────
    form_col, result_col = st.columns([1, 1], gap="large")

    with form_col:
        st.markdown('<div class="section-header">📋 Applicant <span>Details</span></div>', unsafe_allow_html=True)

        # Personal Info
        with st.expander("👤 Personal Information", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
                education = st.selectbox("Education", ["Graduate", "Not Graduate"], key="edu")
            with c2:
                married = st.selectbox("Marital Status", ["Yes", "No"], key="married")
                dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"], key="dep")

        # Financial Info
        with st.expander("💰 Financial Information", expanded=True):
            applicant_income = st.number_input(
                "Applicant Monthly Income (₹)",
                min_value=0, max_value=100000,
                value=5000, step=500,
                help="Your monthly take-home income in ₹"
            )
            coapplicant_income = st.number_input(
                "Co-Applicant Monthly Income (₹)",
                min_value=0, max_value=50000,
                value=0, step=500,
                help="Co-applicant income (enter 0 if none)"
            )
            loan_amount = st.number_input(
                "Loan Amount (in ₹ Thousands)",
                min_value=10, max_value=700,
                value=150, step=10,
                help="Loan amount in thousands — e.g., 150 means ₹1,50,000"
            )

        # Loan Details
        with st.expander("🏷️ Loan Details", expanded=True):
            loan_term = st.selectbox(
                "Loan Term (months)",
                [12, 36, 60, 84, 120, 180, 240, 300, 360, 480],
                index=9,
                help="Duration to repay the loan"
            )
            credit_history = st.radio(
                "Credit History",
                [1, 0],
                format_func=lambda x: "✅ Good (All dues cleared)" if x == 1 else "❌ Bad (Dues pending)",
                horizontal=True,
                help="1 = All loans repaid; 0 = Outstanding dues"
            )
            self_employed = st.selectbox("Self Employed?", ["No", "Yes"])
            property_area = st.selectbox(
                "Property Area",
                ["Urban", "Semiurban", "Rural"],
                help="Location of the property to be purchased"
            )

        # Computed quick summary
        total_income = applicant_income + coapplicant_income
        emi = (loan_amount * 1000) / (loan_term if loan_term > 0 else 1)
        income_after_emi = total_income - emi
        st.markdown(f"""
        <div class="info-box">
        💡 <b>Quick Summary</b><br>
        Total Income: <b>₹{total_income:,.0f}/mo</b> &nbsp;|&nbsp;
        Estimated EMI: <b>₹{emi:,.0f}/mo</b> &nbsp;|&nbsp;
        Balance after EMI: <b style='color:{"#10b981" if income_after_emi>0 else "#ef4444"}'>₹{income_after_emi:,.0f}/mo</b>
        </div>
        """, unsafe_allow_html=True)

        predict_btn = st.button("🔍 Predict Loan Approval", use_container_width=True)

    # ── Result Panel ──────────────────────────
    with result_col:
        st.markdown('<div class="section-header">📊 Prediction <span>Result</span></div>', unsafe_allow_html=True)

        if predict_btn:
            if not model_loaded:
                st.error("Model not loaded. Please train the model first.")
            else:
                input_data = {
                    "gender": gender, "married": married,
                    "dependents": dependents, "education": education,
                    "self_employed": self_employed,
                    "applicant_income": applicant_income,
                    "coapplicant_income": coapplicant_income,
                    "loan_amount": loan_amount,
                    "loan_term": loan_term,
                    "credit_history": credit_history,
                    "property_area": property_area,
                }

                with st.spinner("Analyzing application…"):
                    import time; time.sleep(0.6)   # small delay for UX
                    X_input = preprocess_input(input_data, feature_columns, scaler)
                    prediction  = model.predict(X_input)[0]
                    probability = model.predict_proba(X_input)[0]

                approved = (prediction == 1)
                conf = probability[1] if approved else probability[0]
                conf_pct = round(conf * 100, 1)

                if approved:
                    st.markdown(f"""
                    <div class="result-approved">
                        <span class="result-icon">✅</span>
                        <div class="result-title">LOAN APPROVED!</div>
                        <div class="result-subtitle">Congratulations! Based on the provided details, this application meets the approval criteria.</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-rejected">
                        <span class="result-icon">❌</span>
                        <div class="result-title">LOAN REJECTED</div>
                        <div class="result-subtitle">Based on the provided details, this application does not meet the current approval criteria.</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Probability bars
                st.markdown('<div class="section-header" style="font-size:1rem">Confidence <span>Scores</span></div>', unsafe_allow_html=True)
                p_approve = round(probability[1] * 100, 1)
                p_reject  = round(probability[0] * 100, 1)

                st.markdown(f"""
                <div class="prob-bar-container">
                    <div class="prob-label"><span>✅ Approval Probability</span><span><b style="color:#10b981">{p_approve}%</b></span></div>
                    <div class="prob-bar-bg"><div class="prob-bar-fill-green" style="width:{p_approve}%"></div></div>
                </div>
                <div class="prob-bar-container" style="margin-top:0.8rem">
                    <div class="prob-label"><span>❌ Rejection Probability</span><span><b style="color:#ef4444">{p_reject}%</b></span></div>
                    <div class="prob-bar-bg"><div class="prob-bar-fill-red" style="width:{p_reject}%"></div></div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Key factors
                st.markdown('<div class="section-header" style="font-size:1rem">📌 Key <span>Factors</span></div>', unsafe_allow_html=True)
                factors = []
                if credit_history == 1: factors.append(("✅", "Good credit history", "positive"))
                else:                   factors.append(("⚠️", "Poor credit history — major risk", "negative"))
                if total_income > loan_amount * 10: factors.append(("✅", "Strong income-to-loan ratio", "positive"))
                else:                                factors.append(("⚠️", "Low income relative to loan", "negative"))
                if income_after_emi > 0: factors.append(("✅", f"Positive balance after EMI (₹{income_after_emi:,.0f})", "positive"))
                else:                    factors.append(("❌", "Income insufficient to cover EMI", "negative"))
                if education == "Graduate": factors.append(("✅", "Graduate education", "positive"))
                if property_area == "Semiurban": factors.append(("✅", "Semiurban property (higher approval rate)", "positive"))

                for icon, text, kind in factors:
                    color = "#10b981" if kind == "positive" else "#f87171"
                    st.markdown(f"<div style='padding:0.4rem 0; color:{color}; font-size:0.88rem'>{icon} {text}</div>", unsafe_allow_html=True)

                # Application summary
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-header" style="font-size:1rem">📄 Application <span>Summary</span></div>', unsafe_allow_html=True)
                summary_data = {
                    "Field": ["Total Income", "Loan Amount", "Loan Term", "Est. EMI", "Credit History", "Property Area"],
                    "Value": [
                        f"₹{total_income:,}/mo",
                        f"₹{loan_amount*1000:,}",
                        f"{loan_term} months",
                        f"₹{emi:,.0f}/mo",
                        "Good ✅" if credit_history == 1 else "Bad ❌",
                        property_area,
                    ]
                }
                st.dataframe(
                    pd.DataFrame(summary_data),
                    hide_index=True,
                    use_container_width=True
                )

        else:
            st.markdown("""
            <div style='text-align:center; padding:4rem 2rem; color:#334155'>
                <div style='font-size:4rem; margin-bottom:1rem'>🏦</div>
                <div style='font-family:Syne,sans-serif; font-size:1.2rem; color:#475569'>
                    Fill in the applicant<br>details and click <b style='color:#60a5fa'>Predict</b>
                </div>
                <div style='margin-top:1rem; font-size:0.82rem; color:#334155'>
                    The model will analyze the data<br>and return a prediction in under 1 second.
                </div>
            </div>
            """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#         PAGE 2 · DATA ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif "Analysis" in page:

    st.markdown("""
    <div class="hero-header">
        <div class="hero-badge">EXPLORATORY DATA ANALYSIS</div>
        <div class="hero-title">Dataset <span>Insights</span></div>
        <div class="hero-subtitle">Visual analysis of the loan dataset — distributions, correlations, and feature importance.</div>
    </div>
    """, unsafe_allow_html=True)

    df = load_dataset()

    if df is None:
        st.markdown("""
        <div class="info-box">
        ⚠️ Dataset not found. Please place your CSV file in the <code>dataset/</code> folder
        (filename: <code>loan_data.csv</code> or <code>train.csv</code>) and refresh.
        </div>
        """, unsafe_allow_html=True)
    else:
        set_chart_style()

        # ── Dataset Overview ──────────────────
        st.markdown('<div class="section-header">📋 Dataset <span>Overview</span></div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", len(df))
        c2.metric("Features", df.shape[1])
        approved_col = "Loan_Status" if "Loan_Status" in df.columns else df.columns[-1]
        if approved_col in df.columns:
            n_approved = (df[approved_col].str.upper() == "Y").sum() if df[approved_col].dtype == object else (df[approved_col] == 1).sum()
            c3.metric("Approved", int(n_approved))
            c4.metric("Rejected", int(len(df) - n_approved))

        with st.expander("🔍 Raw Data Preview", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Chart Grid ────────────────────────
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Approval Distribution", "💰 Income Analysis", "📈 Credit History", "🔥 Correlation"])

        # Tab 1: Approval Distribution
        with tab1:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.patch.set_facecolor("#0f172a")

            if approved_col in df.columns:
                status_counts = df[approved_col].value_counts()
                colors = ["#10b981", "#ef4444"]
                axes[0].pie(
                    status_counts.values,
                    labels=["Approved" if str(s).upper() == "Y" else "Rejected" for s in status_counts.index],
                    colors=colors, autopct="%1.1f%%",
                    startangle=140,
                    textprops={"color": "#e2e8f0", "fontsize": 11},
                    wedgeprops={"edgecolor": "#0f172a", "linewidth": 2}
                )
                axes[0].set_title("Loan Approval Split", color="#e2e8f0", fontsize=13, fontweight="bold")

            # By Education
            if "Education" in df.columns and approved_col in df.columns:
                edu_status = pd.crosstab(df["Education"], df[approved_col])
                edu_status.plot(kind="bar", ax=axes[1], color=["#ef4444", "#10b981"],
                                edgecolor="#0f172a", linewidth=1.5)
                axes[1].set_title("Approval by Education", color="#e2e8f0", fontsize=13, fontweight="bold")
                axes[1].set_xlabel("Education Level", color="#94b4d4")
                axes[1].set_ylabel("Count", color="#94b4d4")
                axes[1].tick_params(axis="x", rotation=0)
                axes[1].legend(["Rejected", "Approved"], labelcolor="#e2e8f0", facecolor="#0f172a",
                                edgecolor="#1e3a5f")

            plt.tight_layout(pad=2)
            st.pyplot(fig)
            plt.close()

        # Tab 2: Income Analysis
        with tab2:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.patch.set_facecolor("#0f172a")

            if "ApplicantIncome" in df.columns:
                axes[0].hist(df["ApplicantIncome"].dropna(), bins=40, color="#3b82f6",
                             alpha=0.8, edgecolor="#0f172a")
                axes[0].set_title("Applicant Income Distribution", color="#e2e8f0", fontsize=13, fontweight="bold")
                axes[0].set_xlabel("Monthly Income (₹)", color="#94b4d4")
                axes[0].set_ylabel("Frequency", color="#94b4d4")
                axes[0].axvline(df["ApplicantIncome"].median(), color="#f59e0b",
                                linestyle="--", label=f'Median: ₹{df["ApplicantIncome"].median():,.0f}')
                axes[0].legend(labelcolor="#e2e8f0", facecolor="#0f172a", edgecolor="#1e3a5f")

            if "LoanAmount" in df.columns:
                axes[1].hist(df["LoanAmount"].dropna(), bins=40, color="#8b5cf6",
                             alpha=0.8, edgecolor="#0f172a")
                axes[1].set_title("Loan Amount Distribution", color="#e2e8f0", fontsize=13, fontweight="bold")
                axes[1].set_xlabel("Loan Amount (₹ Thousands)", color="#94b4d4")
                axes[1].set_ylabel("Frequency", color="#94b4d4")
                axes[1].axvline(df["LoanAmount"].median(), color="#f59e0b",
                                linestyle="--", label=f'Median: ₹{df["LoanAmount"].median():.0f}K')
                axes[1].legend(labelcolor="#e2e8f0", facecolor="#0f172a", edgecolor="#1e3a5f")

            plt.tight_layout(pad=2)
            st.pyplot(fig)
            plt.close()

        # Tab 3: Credit History
        with tab3:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.patch.set_facecolor("#0f172a")

            if "Credit_History" in df.columns and approved_col in df.columns:
                ch_status = pd.crosstab(df["Credit_History"], df[approved_col])
                ch_status.plot(kind="bar", ax=axes[0], color=["#ef4444", "#10b981"],
                               edgecolor="#0f172a", linewidth=1.5)
                axes[0].set_title("Approval by Credit History", color="#e2e8f0", fontsize=13, fontweight="bold")
                axes[0].set_xticklabels(["Bad Credit", "Good Credit"], rotation=0)
                axes[0].set_ylabel("Count", color="#94b4d4")
                axes[0].legend(["Rejected", "Approved"], labelcolor="#e2e8f0", facecolor="#0f172a", edgecolor="#1e3a5f")

            if "Property_Area" in df.columns and approved_col in df.columns:
                area_status = pd.crosstab(df["Property_Area"], df[approved_col])
                area_status.plot(kind="bar", ax=axes[1], color=["#ef4444", "#10b981"],
                                 edgecolor="#0f172a", linewidth=1.5)
                axes[1].set_title("Approval by Property Area", color="#e2e8f0", fontsize=13, fontweight="bold")
                axes[1].set_xlabel("Property Area", color="#94b4d4")
                axes[1].tick_params(axis="x", rotation=0)
                axes[1].legend(["Rejected", "Approved"], labelcolor="#e2e8f0", facecolor="#0f172a", edgecolor="#1e3a5f")

            plt.tight_layout(pad=2)
            st.pyplot(fig)
            plt.close()

        # Tab 4: Correlation Heatmap
        with tab4:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()
                fig, ax = plt.subplots(figsize=(10, 7))
                fig.patch.set_facecolor("#0f172a")
                sns.heatmap(
                    corr_matrix, ax=ax,
                    annot=True, fmt=".2f",
                    cmap=sns.diverging_palette(230, 20, as_cmap=True),
                    linewidths=0.5, linecolor="#0f172a",
                    annot_kws={"size": 9, "color": "#e2e8f0"},
                    cbar_kws={"shrink": 0.8}
                )
                ax.set_title("Feature Correlation Heatmap", color="#e2e8f0", fontsize=14, fontweight="bold", pad=15)
                ax.tick_params(colors="#94b4d4", labelsize=9)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.info("Not enough numeric columns for a correlation heatmap.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#         PAGE 3 · MODEL INFO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif "Model" in page:

    st.markdown("""
    <div class="hero-header">
        <div class="hero-badge">MACHINE LEARNING CONCEPTS</div>
        <div class="hero-title">Model <span>Explanation</span></div>
        <div class="hero-subtitle">Understand the ML pipeline — from raw data to final prediction.</div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🧠 Why Logistic Regression?", "🔄 ML Pipeline", "📐 Math Behind It"])

    with tab1:
        st.markdown("""
        <div class="section-header">🧠 Why <span>Logistic Regression</span>?</div>
        <div class="info-box">
        <b>Logistic Regression</b> is a supervised machine learning algorithm used for
        <b>binary classification</b> — problems where the output is one of two classes
        (here: <i>Approved = 1</i> or <i>Rejected = 0</i>).
        </div>
        """, unsafe_allow_html=True)

        reasons = [
            ("⚡", "Efficiency", "Fast to train and predict — ideal for real-time loan decisions."),
            ("🔍", "Interpretability", "We can understand which features (credit history, income) drive the prediction — critical for financial decisions."),
            ("📊", "Probability Output", "Gives a probability score (0–100%), not just a yes/no — allows setting custom thresholds."),
            ("🏦", "Industry Standard", "Banks worldwide use logistic regression as a baseline for credit scoring."),
            ("🎓", "Beginner-Friendly", "Clean mathematical foundation — perfect for a B.Tech project to understand deeply."),
        ]
        for icon, title, desc in reasons:
            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.03); border:1px solid rgba(100,160,255,0.1);
                        border-radius:12px; padding:1rem 1.2rem; margin-bottom:0.7rem; display:flex; gap:1rem;'>
                <span style='font-size:1.6rem'>{icon}</span>
                <div>
                    <div style='font-weight:600; color:#e2e8f0; font-size:0.95rem'>{title}</div>
                    <div style='color:#94b4d4; font-size:0.85rem; margin-top:0.2rem'>{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="section-header">🔄 The ML <span>Pipeline</span></div>', unsafe_allow_html=True)
        steps = [
            ("1", "📥 Data Collection", "Load the Kaggle loan dataset (614 rows, 12 columns)"),
            ("2", "🧹 Data Cleaning", "Handle missing values using median (numeric) and mode (categorical)"),
            ("3", "🔢 Encoding", "Convert categorical text to numbers: Male→1, Female→0, etc."),
            ("4", "⚙️ Feature Engineering", "Create new features: TotalIncome, EMI, IncomeAfterEMI"),
            ("5", "✂️ Train/Test Split", "80% training data, 20% testing data (random_state=42)"),
            ("6", "📏 Feature Scaling", "StandardScaler: normalize all features to same range"),
            ("7", "🤖 Model Training", "Fit LogisticRegression on 80% training data"),
            ("8", "📊 Evaluation", "Accuracy, Confusion Matrix, Classification Report"),
            ("9", "💾 Save Model", "Pickle the trained model + scaler for the web app"),
            ("10", "🌐 Deploy", "Streamlit reads the pickle file and runs predictions in real time"),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div style='display:flex; gap:1rem; margin-bottom:0.6rem; align-items:flex-start'>
                <div style='min-width:32px; height:32px; background:rgba(37,99,235,0.3); border:1px solid #3b82f6;
                            border-radius:50%; display:flex; align-items:center; justify-content:center;
                            color:#60a5fa; font-weight:700; font-size:0.85rem'>{num}</div>
                <div>
                    <div style='font-weight:600; color:#e2e8f0; font-size:0.9rem'>{title}</div>
                    <div style='color:#64748b; font-size:0.82rem'>{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="section-header">📐 The <span>Mathematics</span></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        <b>Sigmoid Function</b> — converts any real number to a probability between 0 and 1:
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"\sigma(z) = \frac{1}{1 + e^{-z}}")
        st.markdown("""
        <div class="info-box" style="margin-top:1rem">
        Where <b>z</b> is the linear combination of features:
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_n x_n")
        st.markdown("""
        <div class="info-box" style="margin-top:1rem">
        <b>Decision Rule:</b> If σ(z) ≥ 0.5 → Approved (1), else Rejected (0).<br><br>
        The model <b>learns</b> the β (beta) coefficients during training by
        minimizing the <b>Log Loss</b> (Cross-Entropy Loss) using gradient descent.
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i\log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)\right]")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#         PAGE 4 · ABOUT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif "About" in page:

    st.markdown("""
    <div class="hero-header">
        <div class="hero-badge">PROJECT DOCUMENTATION</div>
        <div class="hero-title">About <span>This Project</span></div>
        <div class="hero-subtitle">Dataset info, features explained, and setup guide.</div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📦 Dataset Info", "🔢 Feature Guide", "🚀 Setup Steps"])

    with tab1:
        st.markdown('<div class="section-header">📦 <span>Dataset</span> Information</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        <b>Source:</b> Kaggle — Dream Housing Finance Loan Prediction Dataset<br>
        <b>Link:</b> <a href="https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset" target="_blank" style="color:#60a5fa">
        kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset</a><br>
        <b>Size:</b> 614 rows × 13 columns<br>
        <b>Target:</b> Loan_Status (Y = Approved, N = Rejected)
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="section-header">🔢 Feature <span>Dictionary</span></div>', unsafe_allow_html=True)
        features_df = pd.DataFrame({
            "Column": ["Loan_ID","Gender","Married","Dependents","Education","Self_Employed",
                       "ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term",
                       "Credit_History","Property_Area","Loan_Status"],
            "Type":   ["ID","Categorical","Categorical","Categorical","Categorical","Categorical",
                       "Numerical","Numerical","Numerical","Numerical",
                       "Binary","Categorical","Target"],
            "Description": [
                "Unique Loan ID",
                "Male / Female",
                "Applicant married? Yes / No",
                "Number of dependents: 0, 1, 2, 3+",
                "Graduate / Not Graduate",
                "Self-employed? Yes / No",
                "Monthly income of applicant (₹)",
                "Monthly income of co-applicant (₹)",
                "Loan amount in thousands (₹)",
                "Term of loan in months",
                "1 = Good (cleared all dues), 0 = Bad",
                "Rural / Semiurban / Urban",
                "Y = Approved, N = Rejected (TARGET)"
            ]
        })
        st.dataframe(features_df, hide_index=True, use_container_width=True)

    with tab3:
        st.markdown('<div class="section-header">🚀 <span>Setup</span> Guide</div>', unsafe_allow_html=True)
        steps_md = """
**Step 1 — Clone or download the project folder**
```bash
cd Desktop
mkdir loan-approval-system && cd loan-approval-system
```

**Step 2 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 3 — Download dataset from Kaggle**
Place the CSV file in the `dataset/` folder and rename it `loan_data.csv`.

**Step 4 — Train the model**
```bash
python model_training.py
```
This generates `loan_model.pkl`.

**Step 5 — Launch the web app**
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`
        """
        st.markdown(steps_md)
