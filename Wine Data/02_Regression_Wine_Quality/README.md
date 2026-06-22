# 📈 Assignment 2 — Regression Analysis (Wine Quality)

Building on the cleaned dataset from Assignment 1, this part investigates how the physicochemical predictors relate to wine **quality**, using simple, multiple, and non-linear regression.

**Institution:** Aristotle University of Thessaloniki (School of Mechanical Engineering, AUTh)
**Course:** Data Analysis — 8th Semester
**Instructor:** Sofia Panagiotidou, Associate Professor
**Author:** Kyriazis Charitopoulos · AEM 7137

---

## ⚙️ Tech Stack

- Python 3
- pandas, NumPy
- statsmodels
- Matplotlib, seaborn

---

## 🏗️ What It Does

**Question 1 — Simple Linear Regression**
Fits an individual `quality = β₀ + β₁·x` model for each predictor, reporting coefficient, standard error, t-statistic, p-value, RSE, R², and F-statistic. Significance is judged at p < 0.05, with regression plots for each significant predictor.

**Question 2 — Multiple Linear Regression**
Fits a full OLS model on all predictors at once, testing each coefficient's significance (H₀: βⱼ = 0) and evaluating overall fit via R², Adjusted R², and the F-statistic.

**Question 3 — Forward Selection**
Greedily builds a model by adding, at each step, the variable that most reduces the residual sum of squares (RSS), stopping when the next variable is no longer significant.

**Question 4 — Non-linear (Polynomial) Regression**
For `residual sugar`, `chlorides`, and `alcohol`, fits linear, quadratic (X²), and cubic (X³) models, choosing the best by Adjusted R² and plotting fitted curves and residuals.

---

## 📁 Key Files

| File | Description |
|------|-------------|
| `Data_2_EN.py` | Simple, multiple, forward-selection, and polynomial regression |
| `report.pdf` / `.docx` / `.tex` | Methodology, tables, plots, and commentary |

> ℹ️ This script is designed to run **after** the Assignment 1 script — it reuses the cleaned `sarxeio` DataFrame and imported libraries.

---

## 🚀 Running

```bash
pip install pandas numpy statsmodels seaborn matplotlib
# run after the Assignment 1 cleaning script
python Data_2_EN.py
```

---

## 📊 Key Results

| Model | Outcome |
|-------|---------|
| Simple regression | All predictors significant **except** chlorides, pH, sulphates |
| Multiple regression | R² = 0.287, Adjusted R² = 0.285 |
| Forward selection | 9 of 11 variables retained; final R² = 0.2865 |
| Polynomial | Cubic best by Adj R², but gains marginal — linear preferred (parsimony) |

`alcohol` showed the strongest, most significant relationship with quality; `volatile acidity` the strongest negative effect.

---

*Academic coursework — Aristotle University of Thessaloniki. Not for resubmission in other academic contexts.*
