# 🍷 Wine Data Analysis — Course Portfolio

A four-part data-analysis pipeline on a wine-production dataset, taking it from raw, messy CSV all the way to tuned machine-learning classifiers. Each assignment builds on the one before it.

**Institution:** Aristotle University of Thessaloniki (School of Mechanical Engineering, AUTh)
**Course:** Data Analysis — 8th Semester
**Instructor:** Sofia Panagiotidou, Associate Professor
**Author:** Kyriazis Charitopoulos · AEM 7137
**Academic Year:** 2025–2026

---

## ⚙️ Tech Stack

- Python 3
- pandas, NumPy
- Matplotlib, seaborn
- scikit-learn
- statsmodels

---

## 🍇 The Dataset

Physicochemical measurements of red and white wines — `fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`, `chlorides`, `free sulfur dioxide`, `total sulfur dioxide`, `density`, `pH`, `sulphates`, `alcohol` — plus a `quality` score and a `wine_type` label (red / white).

The student ID (**AEM = 7137**) is used as the random seed throughout, so every split and sample is fully reproducible.

---

## 🏗️ The Pipeline

**Assignment 1 — Data Cleaning & Exploratory Analysis**
Turns the raw file into a clean, analysis-ready dataset: deduplication, invalid-value detection, type coercion, missing-value imputation (per wine type), outlier detection (IQR & Z-score), clamp transformation, and a full set of boxplots, histograms, and a correlation matrix.

**Assignment 2 — Regression (predicting `quality`)**
Simple linear regression per predictor, full multiple linear regression, forward stepwise selection, and polynomial (non-linear) models — judged on p-values, R², and Adjusted R².

**Assignment 3 — Classification (predicting `wine_type`)**
Discriminative classifiers (Logistic Regression, LDA, QDA, Naïve Bayes, KNN) evaluated with confusion matrices, accuracy/precision/recall/F1, and ROC/AUC curves. K for KNN chosen by 5-fold cross-validation.

**Assignment 4 — Ensemble Tree Methods (predicting `wine_type`)**
Decision Tree, Bagging, Random Forest, and Gradient Boosting — studying the effect of tree depth and number of estimators, then tuning each model with Grid Search + 5-fold cross-validation.

---

## 📁 Repository Structure

| Folder | Contents |
|--------|----------|
| `assignment-1/` | Cleaning & EDA — script, report |
| `assignment-2/` | Regression — script, report |
| `assignment-3/` | Discriminative classification (Parts A & B) — scripts, report |
| `assignment-4/` | Ensemble tree methods — script, report |

Each folder contains the Python script(s), the report (`.docx`, `.tex`, and `.pdf`), and its own `README.md`.

---

## 🚀 Quick Start

**Requirements:** Python 3.8+, pip

```bash
git clone https://github.com/<your-username>/wine-data-analysis.git
cd wine-data-analysis
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
```

Place the dataset CSV in the assignment folder and run the script, e.g.:

```bash
cd assignment-1
python Data1_EN.py
```

---

## 📊 Headline Results

| Assignment | Task | Best Outcome |
|------------|------|--------------|
| 1 | Cleaning | 6,708 → 5,043 clean rows; outliers handled via IQR + Z-score |
| 2 | Regression | Multiple model R² = 0.287; `alcohol` strongest predictor |
| 3 | Classification | KNN best — Accuracy 0.973, AUC 0.973 |
| 4 | Ensembles | Random Forest best — tuned Accuracy 0.9673 |

---

## 👥 Author

| Name |
|------|
| Kyriazis Charitopoulos |

---

*Academic coursework — Aristotle University of Thessaloniki. Not for resubmission in other academic contexts.*
