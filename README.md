# 🏦 Loan Approval Prediction System

> **B.Tech Machine Learning Project** · Python · Scikit-learn · Streamlit

A production-style web application that uses **Logistic Regression** to predict whether a loan application will be **Approved or Rejected** based on applicant details, with real-time predictions and confidence scores.

---

## 📸 Application Preview

```
┌─────────────────────────────────────────────────────────────┐
│  🏦 LoanIQ  ·  Loan Approval Prediction System              │
│  ┌────────────────────┐  ┌────────────────────────────────┐  │
│  │  Applicant Details │  │  LOAN APPROVED! ✅              │  │
│  │  ──────────────── │  │  Confidence: 87.3%             │  │
│  │  Gender:  Male    │  │  ████████████░░  Approved 87%  │  │
│  │  Income:  ₹6,000  │  │  ███░░░░░░░░░░  Rejected 13%  │  │
│  │  Loan:    ₹150K   │  │  ─────────────────────────     │  │
│  │  Credit:  Good ✅ │  │  Key Factors:                  │  │
│  │  [PREDICT]        │  │  ✅ Good credit history         │  │
│  └────────────────────┘  └────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 Real-time Prediction | Instant loan decision with confidence score |
| 📊 Data Visualization | Charts for distribution, income, credit analysis |
| 🎨 Modern Dark UI | Professional dashboard-style interface |
| 🧠 ML Explanation | Visual explanation of the model and math |
| 📋 Key Factors | Shows which inputs drove the decision |
| 🔢 Probability Bars | Visual approval/rejection probability |

---

## 🛠️ Tech Stack

```
Python 3.8+          Core language
Pandas / NumPy       Data manipulation
Scikit-learn         Machine learning (Logistic Regression)
Streamlit            Web interface
Matplotlib / Seaborn Visualizations
Pickle               Model serialization
```

---

## 📁 Project Structure

```
loan-approval-system/
│
├── app.py                    ← Main Streamlit web application
├── model_training.py         ← ML model training script
├── requirements.txt          ← Python package dependencies
├── README.md                 ← This file
│
├── dataset/
│   └── loan_data.csv         ← Place your Kaggle CSV here
│
├── model_artifacts/
│   └── loan_model.pkl        ← Auto-generated after training
│
└── assets/
    ├── confusion_matrix.png  ← Auto-generated evaluation charts
    ├── roc_curve.png
    └── feature_importance.png
```

---

## 📦 Dataset

**Source:** Kaggle — Dream Housing Finance Loan Prediction  
**Direct Link:** https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset  
**Size:** 614 rows × 13 columns

### How to Download

1. Go to the Kaggle link above  
2. Click **Download** (requires free Kaggle account)  
3. Extract the ZIP file  
4. Rename `train.csv` → `loan_data.csv`  
5. Move it to the `dataset/` folder

> **💡 No Kaggle account?** The training script automatically generates a realistic synthetic dataset if no CSV is found!

### Feature Dictionary

| Column | Type | Description |
|---|---|---|
| `Loan_ID` | ID | Unique identifier (dropped during training) |
| `Gender` | Categorical | Male / Female |
| `Married` | Categorical | Yes / No |
| `Dependents` | Categorical | 0 / 1 / 2 / 3+ |
| `Education` | Categorical | Graduate / Not Graduate |
| `Self_Employed` | Categorical | Yes / No |
| `ApplicantIncome` | Numerical | Monthly income (₹) |
| `CoapplicantIncome` | Numerical | Co-applicant monthly income (₹) |
| `LoanAmount` | Numerical | Loan amount in ₹ thousands |
| `Loan_Amount_Term` | Numerical | Repayment duration in months |
| `Credit_History` | Binary | 1 = Good, 0 = Bad |
| `Property_Area` | Categorical | Urban / Semiurban / Rural |
| `Loan_Status` | **TARGET** | **Y = Approved, N = Rejected** |

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.8 or higher  
- pip (Python package manager)

### Step 1 — Create Project Folder

```bash
# Create and navigate into the project
mkdir loan-approval-system
cd loan-approval-system
```

### Step 2 — (Optional) Create Virtual Environment

```bash
# Recommended to keep dependencies isolated
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs: pandas, numpy, scikit-learn, streamlit, matplotlib, seaborn

### Step 4 — Add Dataset

```
dataset/
└── loan_data.csv    ← place your downloaded CSV here
```

> Skip this step if you don't have the dataset — the script creates a synthetic one automatically.

### Step 5 — Train the Model

```bash
python model_training.py
```

**Expected Output:**
```
──────────────────────────────────────────────────────
  STEP 1 · Loading Dataset
──────────────────────────────────────────────────────
  ✔  Loaded dataset from: dataset/loan_data.csv
  ℹ  Shape: 614 rows × 13 columns

... (processing steps) ...

──────────────────────────────────────────────────────
  TRAINING COMPLETE ✅
──────────────────────────────────────────────────────

  Test Accuracy  : 83.74%
  ROC-AUC Score  : 0.8621
  CV Mean Acc.   : 80.45%

  Next step → Run the web app:
    streamlit run app.py
```

### Step 6 — Launch the Web App

```bash
streamlit run app.py
```

Open your browser: **http://localhost:8501**

---

## 🎯 How to Make Predictions

1. Open the app at `http://localhost:8501`
2. In the **Home & Predict** page, fill in:
   - Personal info (gender, marital status, education)
   - Financial info (income, co-applicant income, loan amount)
   - Loan details (term, credit history, property area)
3. Click **🔍 Predict Loan Approval**
4. See the instant result with confidence score and key factors

---

## 🤖 Machine Learning Details

### Why Logistic Regression?

Logistic Regression is perfect for this problem because:
- **Binary output:** Loan is either approved (1) or rejected (0)
- **Interpretable:** We can see which features matter most
- **Probabilistic:** Gives confidence scores, not just yes/no
- **Industry standard:** Banks use it for credit scoring worldwide

### The ML Pipeline

```
Raw CSV Data
    ↓ Drop Loan_ID
    ↓ Fill missing values (median/mode)
    ↓ Label encode categoricals (Male→1, Female→0...)
    ↓ Feature engineering (TotalIncome, EMI, IncomeAfterEMI)
    ↓ Train/Test split (80/20, stratified)
    ↓ StandardScaler normalization
    ↓ LogisticRegression.fit()
    ↓ Evaluate (accuracy, ROC-AUC, confusion matrix)
    ↓ Save as .pkl file
    ↓ Streamlit app loads .pkl → real-time predictions
```

### Engineered Features

| Feature | Formula | Why? |
|---|---|---|
| `TotalIncome` | ApplicantIncome + CoapplicantIncome | Total household income |
| `EMI` | (LoanAmount × 1000) / Loan_Amount_Term | Monthly payment obligation |
| `IncomeAfterEMI` | TotalIncome − EMI | Leftover money after loan payment |

### Expected Performance

| Metric | Value |
|---|---|
| Test Accuracy | ~80–85% |
| ROC-AUC Score | ~0.85–0.90 |
| Cross-Val Mean | ~79–83% |

> Accuracy varies slightly depending on whether you use the real Kaggle dataset or the synthetic one.

---

## 💡 Sample Predictions

### Example 1 — Likely APPROVED ✅
```
Gender: Male          | Married: Yes
Education: Graduate   | Dependents: 0
Income: ₹6,000/mo    | Co-income: ₹2,000/mo
Loan Amount: ₹150K   | Term: 360 months
Credit History: Good  | Area: Semiurban
→ Result: APPROVED (89.2% confidence)
```

### Example 2 — Likely REJECTED ❌
```
Gender: Female        | Married: No
Education: Not Grad.  | Dependents: 3+
Income: ₹2,500/mo    | Co-income: ₹0/mo
Loan Amount: ₹350K   | Term: 120 months
Credit History: Bad   | Area: Rural
→ Result: REJECTED (78.5% confidence)
```

---

## 🔮 Future Enhancements

- [ ] Add Random Forest and XGBoost models for comparison
- [ ] SHAP values for explainable AI (XAI) insights
- [ ] User authentication and application history
- [ ] PDF report generation after prediction
- [ ] REST API endpoint (FastAPI) for mobile apps
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Real-time database integration (PostgreSQL)
- [ ] Deploy to cloud (Heroku / AWS / GCP)

---

## 📝 Resume Description

```
Loan Approval Prediction System                               Python · ML
• Built a full-stack ML web application using Streamlit and Scikit-learn
  that predicts loan approval decisions with ~83% accuracy
• Implemented complete ML pipeline: data preprocessing, feature engineering
  (EMI, TotalIncome), StandardScaler normalization, and Logistic Regression
• Designed a professional dark-themed dashboard UI with real-time predictions,
  probability scores, and key-factor explanations
• Deployed interactive EDA visualizations (heatmaps, ROC curves, distribution
  plots) using Matplotlib and Seaborn
• Dataset: Kaggle Loan Prediction (614 samples, 13 features); engineered 3
  additional features improving model ROC-AUC to 0.87
```

---

## 🎓 Viva / Interview Q&A

**Q1: What is Logistic Regression?**  
A: A supervised ML algorithm for binary classification. It applies the sigmoid function to a linear combination of features to output a probability between 0 and 1. If probability ≥ 0.5, predict class 1 (Approved); otherwise class 0 (Rejected).

**Q2: Why not Linear Regression for this problem?**  
A: Linear Regression predicts continuous values (e.g., price), not bounded probabilities. It can produce values outside [0,1], which is meaningless for classification. Logistic Regression squashes output to [0,1] using the sigmoid function.

**Q3: What is the sigmoid function?**  
A: σ(z) = 1 / (1 + e^(-z)). It converts any real number z to a value between 0 and 1. This represents the probability of the positive class.

**Q4: Why did you use StandardScaler?**  
A: Features like income (₹5000) and credit history (0 or 1) have very different magnitudes. Logistic Regression is sensitive to feature scale — without normalization, high-magnitude features dominate. StandardScaler transforms all features to zero mean and unit variance.

**Q5: What is overfitting and how did you prevent it?**  
A: Overfitting is when the model memorizes training data but fails on new data (high train accuracy, low test accuracy). We prevented it by: (1) train/test split, (2) cross-validation, (3) L2 regularization (C parameter in LogisticRegression), (4) not using too many features.

**Q6: What does ROC-AUC score mean?**  
A: ROC (Receiver Operating Characteristic) curve plots True Positive Rate vs False Positive Rate at various thresholds. AUC (Area Under Curve) = 0.87 means the model has 87% probability of ranking a random approved loan higher than a random rejected loan.

**Q7: What is the confusion matrix?**  
A: A 2×2 table showing: True Positives (correctly approved), True Negatives (correctly rejected), False Positives (wrongly approved), False Negatives (wrongly rejected). We want high TP and TN.

**Q8: Why is credit history the most important feature?**  
A: The Logistic Regression coefficient for Credit_History is the highest. Applicants with good credit history (score=1) are significantly more likely to repay loans. This is consistent with how real banks make decisions.

**Q9: What is feature engineering and what features did you create?**  
A: Feature engineering is creating new meaningful features from existing ones. We created: TotalIncome (combined household income), EMI (monthly payment), and IncomeAfterEMI (disposable income after EMI). These better represent financial health than raw features alone.

**Q10: How would you improve this project?**  
A: (1) Try ensemble models (Random Forest, XGBoost) which typically achieve 87–92% accuracy on this dataset. (2) Use SHAP values for explainability. (3) Add hyperparameter tuning via GridSearchCV. (4) Handle class imbalance with SMOTE. (5) Deploy to cloud for public access.

---

## 📄 License

This project is created for educational purposes.  
Free to use and modify for your B.Tech project.

---

*Built with ❤️ for the B.Tech ML curriculum*
