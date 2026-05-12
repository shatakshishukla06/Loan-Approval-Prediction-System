"""
╔══════════════════════════════════════════════════════════════╗
║         LOAN APPROVAL PREDICTION SYSTEM                      ║
║         Model Training Script                                ║
╚══════════════════════════════════════════════════════════════╝

FILE: model_training.py
PURPOSE: Load the dataset, preprocess it, train a Logistic Regression
         model, evaluate it, and save the artifacts for the web app.

HOW TO RUN:
    python model_training.py

OUTPUT:
    model_artifacts/loan_model.pkl  ← used by app.py
"""

# ─────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────
import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PRETTY PRINT HELPERS
# ─────────────────────────────────────────────
BLUE   = "\033[94m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def section(title):
    print(f"\n{BOLD}{BLUE}{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}{RESET}")

def success(msg):  print(f"  {GREEN}✔  {msg}{RESET}")
def info(msg):     print(f"  {YELLOW}ℹ  {msg}{RESET}")
def warn(msg):     print(f"  {RED}⚠  {msg}{RESET}")


# ══════════════════════════════════════════════
#  STEP 1 · LOAD DATASET
# ══════════════════════════════════════════════
section("STEP 1 · Loading Dataset")

# Try common filenames people use when downloading from Kaggle
POSSIBLE_PATHS = [
    "dataset/loan_data.csv",
    "dataset/train.csv",
    "dataset/loan_sanction_train.csv",
    "dataset/Loan_sanction_train.csv",
]

df = None
for path in POSSIBLE_PATHS:
    if os.path.exists(path):
        df = pd.read_csv(path)
        success(f"Loaded dataset from: {path}")
        break

# ── Fallback: generate a synthetic dataset so the script always works ──
if df is None:
    warn("No CSV found in dataset/ folder. Generating a synthetic demo dataset…")
    warn("For real results, download from: https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset")
    warn("Place the file at: dataset/loan_data.csv")
    print()

    np.random.seed(42)
    N = 614

    # Replicate the statistical properties of the real Kaggle dataset
    genders     = np.random.choice(["Male", "Female"], N, p=[0.81, 0.19])
    married     = np.random.choice(["Yes", "No"],      N, p=[0.65, 0.35])
    dependents  = np.random.choice(["0", "1", "2", "3+"], N, p=[0.57, 0.17, 0.16, 0.10])
    education   = np.random.choice(["Graduate", "Not Graduate"], N, p=[0.78, 0.22])
    self_emp    = np.random.choice(["Yes", "No"],      N, p=[0.14, 0.86])
    app_income  = np.random.exponential(scale=5000, size=N).astype(int) + 1500
    coapp_inc   = np.random.exponential(scale=1500, size=N).astype(int)
    coapp_inc[np.random.rand(N) < 0.40] = 0   # 40% no co-applicant
    loan_amt    = np.random.normal(146, 85, N).clip(9, 700).astype(int)
    loan_term   = np.random.choice([12, 36, 60, 84, 120, 180, 240, 300, 360, 480], N,
                                    p=[0.01, 0.01, 0.02, 0.02, 0.04, 0.08, 0.04, 0.04, 0.69, 0.05])
    credit_hist = np.random.choice([1.0, 0.0, np.nan], N, p=[0.72, 0.16, 0.12])
    prop_area   = np.random.choice(["Urban", "Semiurban", "Rural"], N, p=[0.36, 0.38, 0.26])

    # Realistic loan approval logic
    score = (
        (credit_hist == 1).astype(float) * 2.5
        + (np.array(app_income) > 4000) * 0.6
        + (np.array([d.replace("+","") for d in dependents]).astype(int) < 2) * 0.3
        + (np.array(education) == "Graduate") * 0.4
        + (np.array(prop_area) == "Semiurban") * 0.3
        + np.random.normal(0, 0.4, N)
    )
    loan_status = np.where(score > 2.0, "Y", "N")

    # Add a few missing values (realistic)
    for col_arr in [app_income, loan_amt]:
        idx = np.random.choice(N, size=int(N * 0.03), replace=False)
        col_arr = col_arr.astype(float)
        col_arr[idx] = np.nan

    df = pd.DataFrame({
        "Loan_ID":           [f"LP{str(i).zfill(6)}" for i in range(N)],
        "Gender":            genders,
        "Married":           married,
        "Dependents":        dependents,
        "Education":         education,
        "Self_Employed":     self_emp,
        "ApplicantIncome":   app_income,
        "CoapplicantIncome": coapp_inc,
        "LoanAmount":        loan_amt.astype(float),
        "Loan_Amount_Term":  loan_term.astype(float),
        "Credit_History":    credit_hist,
        "Property_Area":     prop_area,
        "Loan_Status":       loan_status,
    })

    os.makedirs("dataset", exist_ok=True)
    df.to_csv("dataset/loan_data.csv", index=False)
    success("Synthetic dataset saved to dataset/loan_data.csv")

info(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
info(f"Columns: {list(df.columns)}")


# ══════════════════════════════════════════════
#  STEP 2 · INITIAL EXPLORATION
# ══════════════════════════════════════════════
section("STEP 2 · Exploratory Data Analysis")

# Identify the target column (handles both naming conventions)
TARGET_COL = "Loan_Status" if "Loan_Status" in df.columns else df.columns[-1]
info(f"Target column: '{TARGET_COL}'")

# Class distribution
status_counts = df[TARGET_COL].value_counts()
info(f"Class distribution:\n{status_counts.to_string()}")

# Missing values
missing = df.isnull().sum()
missing = missing[missing > 0]
if len(missing):
    info(f"Missing values found:\n{missing.to_string()}")
else:
    success("No missing values!")


# ══════════════════════════════════════════════
#  STEP 3 · DATA PREPROCESSING
# ══════════════════════════════════════════════
section("STEP 3 · Data Preprocessing")

# ── 3a. Drop ID column (not a feature) ────────
if "Loan_ID" in df.columns:
    df.drop(columns=["Loan_ID"], inplace=True)
    success("Dropped 'Loan_ID' column")

# ── 3b. Handle missing values ─────────────────
#   Numeric → fill with MEDIAN  (robust to outliers)
#   Categorical → fill with MODE (most frequent value)
numeric_cols     = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

for col in numeric_cols:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        success(f"Filled '{col}' NaN with median={median_val:.1f}")

# Extra safety: fill any remaining NaN with column median/0
df = df.fillna(df.median(numeric_only=True))
df = df.fillna(0)

for col in categorical_cols:
    if col == TARGET_COL:
        continue
    if df[col].isnull().any():
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        success(f"Filled '{col}' NaN with mode='{mode_val}'")

success("All missing values handled")

# ── 3c. Label Encoding ────────────────────────
# Convert categorical text → integers
# IMPORTANT: these exact same mappings must be used in app.py → preprocess_input()

encoding_maps = {
    "Gender":        {"Male": 1, "Female": 0},
    "Married":       {"Yes": 1, "No": 0},
    "Education":     {"Graduate": 1, "Not Graduate": 0},
    "Self_Employed": {"Yes": 1, "No": 0},
    "Property_Area": {"Urban": 2, "Semiurban": 1, "Rural": 0},
    "Dependents":    {"0": 0, "1": 1, "2": 2, "3+": 3},
}

for col, mapping in encoding_maps.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)
        success(f"Encoded '{col}': {mapping}")

# Encode target: Y → 1, N → 0
# Encode target robustly regardless of dtype
if True:
    df[TARGET_COL] = df[TARGET_COL].astype(str).str.upper().map({"Y": 1, "N": 0}).fillna(0).astype(int)
    success(f"Encoded '{TARGET_COL}': Y→1, N→0")


# ══════════════════════════════════════════════
#  STEP 4 · FEATURE ENGINEERING
# ══════════════════════════════════════════════
section("STEP 4 · Feature Engineering")

# Create new meaningful features that the model can use
# These MUST also be created in app.py → preprocess_input()

df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
success("Created 'TotalIncome' = ApplicantIncome + CoapplicantIncome")

df["EMI"] = (df["LoanAmount"] * 1000) / df["Loan_Amount_Term"].replace(0, np.nan).fillna(1)
success("Created 'EMI' = (LoanAmount × 1000) / Loan_Amount_Term")

df["IncomeAfterEMI"] = df["TotalIncome"] - df["EMI"]
success("Created 'IncomeAfterEMI' = TotalIncome - EMI")

info(f"New shape after feature engineering: {df.shape}")


# ══════════════════════════════════════════════
#  STEP 5 · TRAIN/TEST SPLIT
# ══════════════════════════════════════════════
section("STEP 5 · Train / Test Split")

FEATURES = [col for col in df.columns if col != TARGET_COL]
X = df[FEATURES]
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,     # 80% train, 20% test
    random_state=42,    # reproducibility
    stratify=y          # maintain class balance in both splits
)

info(f"Feature columns: {FEATURES}")
info(f"Training samples: {len(X_train)}")
info(f"Testing  samples: {len(X_test)}")
info(f"Class balance in train: {dict(y_train.value_counts())}")


# ══════════════════════════════════════════════
#  STEP 6 · FEATURE SCALING
# ══════════════════════════════════════════════
section("STEP 6 · Feature Scaling (StandardScaler)")

# StandardScaler: x_scaled = (x - mean) / std
# This ensures no feature dominates just because of its magnitude (income vs credit history)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit on TRAIN only
X_test_scaled  = scaler.transform(X_test)         # only transform test (no leakage)

success("StandardScaler fitted on training data")
success("Test data transformed using same scaler (no data leakage)")


# ══════════════════════════════════════════════
#  STEP 7 · MODEL TRAINING
# ══════════════════════════════════════════════
section("STEP 7 · Training Logistic Regression Model")

model = LogisticRegression(
    C=1.0,                # Regularization strength (1.0 = default)
    solver="lbfgs",       # Optimization algorithm — good for small datasets
    max_iter=1000,        # Maximum iterations for convergence
    random_state=42,
    class_weight="balanced",  # Handle class imbalance automatically
)

model.fit(X_train_scaled, y_train)
success("Model training complete!")

# Show learned coefficients
coef_df = pd.DataFrame({
    "Feature":     FEATURES,
    "Coefficient": model.coef_[0],
}).sort_values("Coefficient", ascending=False)

info("Top positive predictors (push toward APPROVED):")
for _, row in coef_df[coef_df["Coefficient"] > 0].head(5).iterrows():
    print(f"     {row['Feature']:30s} {row['Coefficient']:+.4f}")
info("Top negative predictors (push toward REJECTED):")
for _, row in coef_df[coef_df["Coefficient"] < 0].tail(5).iterrows():
    print(f"     {row['Feature']:30s} {row['Coefficient']:+.4f}")


# ══════════════════════════════════════════════
#  STEP 8 · MODEL EVALUATION
# ══════════════════════════════════════════════
section("STEP 8 · Model Evaluation")

y_pred       = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Core metrics
train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
test_acc  = accuracy_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_pred_proba)

print(f"\n  {'Metric':<30}{'Value':>10}")
print(f"  {'─'*40}")
print(f"  {'Training Accuracy':<30}{train_acc*100:>9.2f}%")
print(f"  {'Test Accuracy':<30}{test_acc*100:>9.2f}%")
print(f"  {'ROC-AUC Score':<30}{roc_auc:>10.4f}")

# Cross-validation for robust accuracy estimate
cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5, scoring="accuracy")
print(f"\n  5-Fold Cross-Validation Scores: {[round(s, 4) for s in cv_scores]}")
print(f"  Mean CV Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
info(f"\nConfusion Matrix:\n{cm}")
print(f"""
  True Negatives  (Correctly Rejected): {cm[0][0]}
  False Positives (Incorrectly Approved): {cm[0][1]}
  False Negatives (Incorrectly Rejected): {cm[1][0]}
  True Positives  (Correctly Approved):  {cm[1][1]}
""")

# Classification Report
print(f"\n{BOLD}  Classification Report:{RESET}")
print(classification_report(y_test, y_pred, target_names=["Rejected", "Approved"]))


# ══════════════════════════════════════════════
#  STEP 9 · SAVE VISUALIZATIONS
# ══════════════════════════════════════════════
section("STEP 9 · Saving Training Visualizations")

os.makedirs("assets", exist_ok=True)
plt.style.use("dark_background")
plt.rcParams.update({"figure.facecolor": "#0f172a", "axes.facecolor": "#0f1e35"})

# ── Confusion Matrix ──
fig, ax = plt.subplots(figsize=(6, 5))
fig.patch.set_facecolor("#0f172a")
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Rejected", "Approved"],
    yticklabels=["Rejected", "Approved"],
    ax=ax, linewidths=1, linecolor="#0f172a",
    annot_kws={"size": 14, "weight": "bold"}
)
ax.set_title("Confusion Matrix", color="#e2e8f0", fontsize=14, pad=12)
ax.set_xlabel("Predicted", color="#94b4d4")
ax.set_ylabel("Actual", color="#94b4d4")
ax.tick_params(colors="#94b4d4")
plt.tight_layout()
plt.savefig("assets/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
success("Saved assets/confusion_matrix.png")

# ── ROC Curve ──
fpr, tpr, _ = roc_curve(y_test, y_pred_proba, pos_label=1)
fig, ax = plt.subplots(figsize=(6, 5))
fig.patch.set_facecolor("#0f172a")
ax.plot(fpr, tpr, color="#3b82f6", lw=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], color="#475569", lw=1, linestyle="--", label="Random Classifier")
ax.fill_between(fpr, tpr, alpha=0.1, color="#3b82f6")
ax.set_title("ROC Curve", color="#e2e8f0", fontsize=14, pad=12)
ax.set_xlabel("False Positive Rate", color="#94b4d4")
ax.set_ylabel("True Positive Rate", color="#94b4d4")
ax.tick_params(colors="#94b4d4")
ax.legend(labelcolor="#e2e8f0", facecolor="#0f172a", edgecolor="#1e3a5f")
ax.grid(True, alpha=0.2, color="#1e3a5f")
plt.tight_layout()
plt.savefig("assets/roc_curve.png", dpi=150, bbox_inches="tight")
plt.close()
success("Saved assets/roc_curve.png")

# ── Feature Importance (Coefficients) ──
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor("#0f172a")
colors_bar = ["#10b981" if c > 0 else "#ef4444" for c in coef_df["Coefficient"]]
ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors_bar, edgecolor="#0f172a")
ax.set_title("Feature Coefficients (Logistic Regression)", color="#e2e8f0", fontsize=13, pad=12)
ax.set_xlabel("Coefficient Value", color="#94b4d4")
ax.axvline(0, color="#475569", lw=1, linestyle="--")
ax.tick_params(colors="#94b4d4")
ax.grid(True, alpha=0.2, axis="x", color="#1e3a5f")
plt.tight_layout()
plt.savefig("assets/feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
success("Saved assets/feature_importance.png")


# ══════════════════════════════════════════════
#  STEP 10 · SAVE MODEL ARTIFACTS
# ══════════════════════════════════════════════
section("STEP 10 · Saving Model Artifacts")

os.makedirs("model_artifacts", exist_ok=True)

artifacts = {
    "model":           model,      # Trained LogisticRegression
    "scaler":          scaler,     # Fitted StandardScaler
    "feature_columns": FEATURES,   # List of feature names in correct order
    "accuracy":        test_acc,   # Test accuracy for display in app
    "roc_auc":         roc_auc,    # ROC-AUC score
    "cv_mean":         cv_scores.mean(),
    "encoding_maps":   encoding_maps,
}

with open("model_artifacts/loan_model.pkl", "wb") as f:
    pickle.dump(artifacts, f)

success("Model saved → model_artifacts/loan_model.pkl")


# ══════════════════════════════════════════════
#  FINAL SUMMARY
# ══════════════════════════════════════════════
section("TRAINING COMPLETE ✅")
print(f"""
  {GREEN}{BOLD}Model is ready!{RESET}

  Test Accuracy  : {GREEN}{test_acc*100:.2f}%{RESET}
  ROC-AUC Score  : {GREEN}{roc_auc:.4f}{RESET}
  CV Mean Acc.   : {GREEN}{cv_scores.mean()*100:.2f}%{RESET}

  Next step → Run the web app:
  {YELLOW}  streamlit run app.py{RESET}

  Then open: {BLUE}http://localhost:8501{RESET}
""")
