# 🍷 Assignment 3 — Classification: Predicting Wine Type

Switching from regression to classification: which predictors best estimate whether a wine is **red or white**, using a family of discriminative classifiers and ROC/AUC evaluation.

**Institution:** Aristotle University of Thessaloniki (School of Mechanical Engineering, AUTh)
**Course:** Data Analysis — 8th Semester
**Instructor:** Sofia Panagiotidou, Associate Professor
**Author:** Kyriazis Charitopoulos · AEM 7137

---

## ⚙️ Tech Stack

- Python 3
- pandas, NumPy
- scikit-learn
- Matplotlib, seaborn

---

## 🏗️ What It Does

Target: `wine_type`, encoded as **red = 0, white = 1**. The `quality` column is excluded from the predictors. Data is split 70% / 30% (stratified) using **AEM = 7137** as the seed.

**Part A — Single-Predictor Classification**
Trains Logistic Regression, LDA, and Naïve Bayes on each of three predictors separately (`alcohol`, `pH`, `chlorides`), comparing confusion matrices, accuracy, precision, recall, and F1 to find the most informative single feature.

**Part B — Full Multi-Predictor Classification**
Uses all predictors with five methods — KNN, Logistic Regression, LDA, QDA, and Naïve Bayes. Adds feature scaling (`StandardScaler`) for distance/optimisation-based models, selects K for KNN via 5-fold cross-validation, and evaluates with ROC curves and AUC on top of the Part A metrics.

---

## 📁 Key Files

| File | Description |
|------|-------------|
| `ergasia_EN.py` | Part A — single-predictor LR / LDA / NB |
| `ml_test_EN.py` | Part B — KNN / LR / LDA / QDA / NB with scaling, CV, and ROC/AUC |
| `report.pdf` / `.docx` / `.tex` | Methodology, confusion matrices, metric tables, ROC curves |
| `Data_Analysis_2026_3rd_Case_Data.csv` | Input dataset (place alongside the scripts) |

---

## 🚀 Running

```bash
pip install pandas numpy scikit-learn seaborn matplotlib
python ergasia_EN.py   # Part A
python ml_test_EN.py   # Part B
```

---

## 📊 Key Results

**Dataset:** 5,295 rows → 3,706 train / 1,589 test.

**Part A:** `chlorides` was the most informative single predictor (best accuracy across all three methods); Naïve Bayes edged out LDA (Accuracy ≈ 0.782, F1 ≈ 0.871).

**Part B (all predictors):**

| Method | Accuracy | AUC |
|--------|----------|-----|
| **KNN** (Best K = 7) | **0.973** | **0.973** |
| Logistic Regression | 0.971 | 0.966 |
| LDA | 0.972 | 0.966 |
| Naïve Bayes | 0.948 | 0.955 |
| QDA | 0.925 | 0.942 |

KNN came out on top; KNN, Logistic Regression, and LDA were all closely matched.

---

*Academic coursework — Aristotle University of Thessaloniki. Not for resubmission in other academic contexts.*
