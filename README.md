# Risk Alert Classifier

**Institution:** RED & WHITE SKILL EDUCATION  
**Domain:** Supervised Learning — Binary Classification  
**Project Duration:** 6 Hours

---

## Overview

This project builds a machine learning pipeline to classify bank customers as **High Risk (1)** or **Low Risk (0)** based on financial and behavioural attributes. The goal is to help banks proactively identify customers likely to default or commit fraud, thereby minimising unrecoverable financial losses.

The project covers the full ML lifecycle — data preparation, imbalance handling, model training, hyperparameter tuning, and ROC-based evaluation — using a real-world style dataset of 4,600 customer records.

---

## Dataset

| Property | Details |
|---|---|
| **File** | `Risk_Alert_Classifier_Dataset_4600.xlsx` |
| **Rows** | 4,600 |
| **Features** | 18 input columns + 1 target |
| **Target Variable** | `risk_status` (0 = Low Risk, 1 = High Risk) |
| **Class Distribution** | Low Risk: 4,043 (87.9%) — High Risk: 557 (12.1%) |
| **Imbalance Ratio** | ~7.3 : 1 |

### Features

| Column | Type | Description |
|---|---|---|
| `customer_id` | int | Unique customer identifier (dropped during training) |
| `age` | float | Customer age (has missing values) |
| `gender` | object | Gender of the customer |
| `region` | object | Geographic region (has missing values) |
| `employment_type` | object | Employment category (has missing values) |
| `annual_income_inr` | float | Annual income in INR (has missing values) |
| `credit_score` | float | Credit score (has missing values) |
| `credit_utilization_ratio` | float | Ratio of credit used to total credit (has missing values) |
| `missed_payments_12m` | int | Number of missed payments in the last 12 months |
| `avg_late_payment_days` | float | Average days of payment delay |
| `monthly_transaction_count` | int | Number of transactions per month |
| `monthly_spend_inr` | float | Monthly spending in INR (has missing values) |
| `cash_advance_count_6m` | int | Cash advances taken in the last 6 months |
| `complaints_last_6m` | int | Number of complaints filed in the last 6 months |
| `failed_login_attempts_3m` | int | Failed login attempts in the last 3 months |
| `account_tenure_months` | int | Duration of account in months |
| `last_transaction_date` | datetime | Date of last transaction (dropped during training) |
| `debt_balance_inr` | int | Outstanding debt in INR |
| `risk_status` | int | **Target** — 0: Low Risk, 1: High Risk |

---

## Project Structure

```
Risk_Alert_Classifier/
│
├── Risk_Alert_Classifier.ipynb          # Main project notebook
├── Risk_Alert_Classifier_Dataset_4600.xlsx  # Dataset
└── README.md                            # Project documentation
```

---

## Methodology

### Part A — Theory
Covers conceptual questions on Logistic Regression, classification metrics (Precision, Recall, F1, AUC-ROC), Type-I vs Type-II errors, and the problem of class imbalance.

### Part B — Data Preparation
- Loaded dataset and identified 18 features + 1 target variable
- Dropped `customer_id` and `last_transaction_date` (non-predictive / date column)
- Encoded categorical features using `LabelEncoder`
- Handled missing values using **KNN Imputer** (k=5) for multivariate imputation
- Applied **stratified 80/20 train-test split** to preserve class ratios
- Scaled features using `StandardScaler` (required for Logistic Regression)

### Part C — Baseline Model (Logistic Regression)
- Trained a baseline Logistic Regression model
- Evaluated using confusion matrix, accuracy, precision, recall, and F1-score
- Identified high False Negative count as the primary concern for the banking use case

### Part D — Imbalance Handling
Applied and compared four balancing techniques on the training data:

| Technique | Effect |
|---|---|
| Random Under-Sampling | Strong Recall improvement; some data loss |
| Random Over-Sampling | Moderate improvement; risk of overfitting on duplicates |
| **SMOTE** | Best Recall-Precision tradeoff; generates synthetic minority samples |
| ADASYN | Similar to SMOTE; adaptive density near decision boundaries |

**SMOTE** was selected as the best balancing strategy for downstream tree models.

### Part E — Tree-Based Models
- **Decision Tree** (untuned): Severe overfitting — training accuracy ≈ 100%, poor generalization
- **Random Forest** (100 estimators): Significantly reduced overfitting gap through ensemble averaging; much stronger baseline

### Part F — Hyperparameter Tuning
- **RandomizedSearchCV** applied to both Decision Tree and Random Forest (5-fold CV, F1 scoring)
- **GridSearchCV** used to fine-tune the Random Forest around the best RandomizedSearch parameters
- Tuned Random Forest delivered the best generalization across all metrics

### Part G — ROC Analysis
- Generated ROC curves for all five models (LR, DT untuned, DT tuned, RF untuned, RF tuned)
- Compared AUC-ROC scores; Tuned Random Forest achieved the highest AUC
- **Primary evaluation metric: Recall** — minimising False Negatives (Type-II errors) is the core business requirement

### Part H — Final Analysis
- Feature importance extracted from Tuned Random Forest
- Top predictors: **credit score, missed payments, credit utilization ratio, debt balance**
- Business recommendations on threshold adjustment and re-training cadence

---

## Results Summary

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression (Baseline) | — | — | — | — | — |
| Decision Tree (Tuned) | — | — | — | — | — |
| **Random Forest (Tuned)** | — | — | — | — | **Best** |

> Exact metric values are printed when the notebook is executed. The Tuned Random Forest ranked first on both Recall and AUC-ROC.

---

## Final Model & Recommendations

**Best Model:** Tuned Random Forest (trained on SMOTE-balanced data, tuned with GridSearchCV)

**Deployment Recommendations:**
- Lower classification threshold to **0.35–0.40** (from default 0.5) to increase Recall and catch more high-risk customers
- Monitor top-4 features most closely: credit score, missed payments, credit utilization ratio, debt balance
- Re-train the model **monthly** as customer risk profiles evolve over time

**Business Rationale:**  
A False Negative (missed high-risk customer) leads to potential loan default or fraud — an unrecoverable loss. A False Positive (low-risk customer flagged) only results in an unnecessary follow-up, which is a recoverable operational cost. Optimising for Recall is therefore the correct business decision.

---

## Tech Stack

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn` | Visualisation |
| `scikit-learn` | ML models, metrics, preprocessing, tuning |
| `imbalanced-learn` | SMOTE, ADASYN, over/under-sampling |

---

## How to Run

1. Clone or download the repository
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn openpyxl
   ```
3. Open the notebook:
   ```bash
   jupyter notebook Risk_Alert_Classifier.ipynb
   ```
4. Run all cells in order (Kernel → Restart & Run All)

> Make sure `Risk_Alert_Classifier_Dataset_4600.xlsx` is in the same directory as the notebook.

---

## Author

**Jatin Kumar**  
B.Tech — Artificial Intelligence & Machine Learning  
RED & WHITE SKILL EDUCATION
