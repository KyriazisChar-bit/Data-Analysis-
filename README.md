# 📊 Data Analysis — Course Portfolio

A five-assignment journey through the full data-science workflow: from cleaning a raw, broken CSV all the way to training convolutional neural networks. Built over one semester, each assignment added a new layer of theory and tooling on top of the last.

**Institution:** Aristotle University of Thessaloniki (School of Mechanical Engineering, AUTh)
**Course:** Data Analysis — 8th Semester
**Instructor:** Sofia Panagiotidou, Associate Professor
**Author:** Kyriazis Charitopoulos · AEM 7137
**Academic Year:** 2025–2026

---

## ⚙️ Tech Stack

- **Language:** Python 3
- **Data handling:** pandas, NumPy
- **Statistics:** statsmodels
- **Machine learning:** scikit-learn
- **Deep learning:** TensorFlow / Keras
- **Visualisation:** Matplotlib, seaborn

---

## 🎯 What This Portfolio Covers

Two datasets, one continuous thread. Assignments 1–4 take a **wine physicochemical dataset** from raw file to tuned classifiers; Assignment 5 is a standalone **MNIST** deep-learning study. Together they walk through every stage a real analysis touches: cleaning, exploring, modelling, evaluating, and tuning.

---

## 🧠 What I Learned

**Data Cleaning & Preprocessing**
Real data is messy. I learned to detect and remove duplicates, spot invalid sentinel values (`-999`, `?`, `9999`), coerce inconsistent types, and reason about *when* to drop a row versus impute it. I imputed missing values by group (per wine type) rather than globally, and used domain knowledge (pH can't exceed 14, alcohol can't exceed 100%) to catch errors a purely statistical check would miss.

**Exploratory Data Analysis**
How to *look* at data before modelling it — descriptive statistics, histograms, boxplots, and correlation heatmaps — and how to read them: spotting skew, multimodality, outliers, and relationships between variables.

**Outlier Detection**
Two complementary methods — the **IQR rule** and **Z-scores** (3.5σ) — and how to act on what they find through clamp/clip transformation rather than blind deletion. I also saw how combining classes inflates outlier counts versus analysing groups separately.

**Statistical Regression & Inference**
The mechanics *and* the interpretation of regression: reading coefficients, standard errors, t-statistics, and **p-values**; testing the null hypothesis H₀: βⱼ = 0; and judging fit with R² and Adjusted R². I learned the difference between simple and multiple regression, the danger of **multicollinearity**, and how **forward stepwise selection** trades model size for predictive gain.

**The Bias–Variance Tradeoff**
Made concrete by watching a decision tree's test error fall then rise as depth increased — underfitting on one side, overfitting on the other — and by seeing why **ensembles** (Bagging, Random Forests) reduce variance by averaging many decorrelated trees.

**Classification & Discriminative Methods**
A whole toolkit — Logistic Regression, LDA, QDA, Naïve Bayes, and KNN — plus the assumptions behind each (e.g. why QDA allows curved boundaries, why KNN needs feature scaling).

**Model Evaluation Done Properly**
Beyond accuracy: confusion matrices, precision, recall, F1, and **ROC curves / AUC** — and crucially, *why* a single number lies. I learned that a model predicting "always white" can look accurate while being useless, and how recall, precision, and AUC tell the fuller story. I also used **k-fold cross-validation** to choose hyperparameters (like K in KNN) honestly.

**Ensemble Learning & Hyperparameter Tuning**
Decision Trees, Bagging, Random Forests, and Gradient Boosting — how they differ, why Random Forest's feature subsampling (`max_features`) decorrelates trees, and how **Grid Search + cross-validation** systematically squeezes out the best configuration. I saw firsthand that strong ensembles are nearly optimal "out of the box," leaving little room for tuning to improve.

**Deep Learning**
Building neural networks from scratch in Keras — dense layers, activations (ReLU, Softmax), optimisers (Adam), and loss functions — then learning *why* a **CNN** beats a fully-connected net on images: weight sharing and translation invariance deliver higher accuracy with **fewer** parameters.

**Reproducibility & Workflow**
Seeding every random operation (with my student ID) so results are repeatable, separating train/test data correctly, and documenting methodology alongside code and results.

---

## 🗺️ The Journey

| # | Assignment | Theme | Key Takeaway |
|---|------------|-------|--------------|
| 1 | Cleaning & EDA | Data preparation | Most of the work is making data trustworthy |
| 2 | Regression | Statistical modelling | Significance, fit, and parsimony |
| 3 | Classification | Discriminative methods | Evaluate with the right metric, not just accuracy |
| 4 | Ensemble Trees | Bias–variance & tuning | Random Forests are robust and hard to beat |
| 5 | Neural Networks | Deep learning | Architecture matters — CNNs are built for images |

---

## 📁 Repository Structure

| Folder | Contents |
|--------|----------|
| `assignment-1/` | Wine cleaning & exploratory analysis |
| `assignment-2/` | Regression (predicting quality) |
| `assignment-3/` | Classification (predicting wine type) |
| `assignment-4/` | Ensemble tree methods |
| `assignment-5/` | MNIST neural networks (Simple NN vs CNN) |

Each folder contains its Python script, the report (`.pdf`), and a dedicated `README.md`.

---

## 🚀 Quick Start

**Requirements:** Python 3.8+, pip

```bash
git clone https://github.com/<your-username>/data-analysis-portfolio.git
cd data-analysis-portfolio
pip install pandas numpy statsmodels scikit-learn tensorflow matplotlib seaborn
```

Then enter any assignment folder, place its dataset alongside the script, and run it.

---

## 👥 Author

| Name | Student ID |
|------|------------|
| Kyriazis Charitopoulos | 7137 |

---

*Academic coursework — Aristotle University of Thessaloniki. Not for resubmission in other academic contexts.*
