# 🌳 Assignment 4 — Ensemble Tree Methods (Wine Type)

Tackling the same red/white classification problem with tree-based ensembles, and studying how depth, the number of estimators, and hyperparameter tuning affect performance.

**Institution:** Aristotle University of Thessaloniki (School of Mechanical Engineering, AUTh)
**Course:** Data Analysis — 8th Semester
**Instructor:** Sofia Panagiotidou, Associate Professor
**Author:** Kyriazis Charitopoulos · AEM 7137

---

## ⚙️ Tech Stack

- Python 3
- pandas, NumPy
- scikit-learn
- Matplotlib

---

## 🏗️ What It Does

Predictors: all variables except `quality` and the target `wine_type` (**p = 12**). Split using **AEM = 7137** as the seed.

**Tasks 1–4 — Base Models**
Trains four classifiers with default-ish settings: a shallow Decision Tree (`max_depth=2`), Bagging (200 estimators), Random Forest (200 estimators, `max_features = p/2`), and Gradient Boosting (200 estimators, `lr=0.1`, `max_depth=1`).

**Task 5 — Depth vs Test Error**
Sweeps a Decision Tree's `max_depth` from 1 to 20, plotting test error to reveal the underfitting → optimum → overfitting curve.

**Task 6 — Estimators vs Test Error**
Varies `n_estimators` from 1 to 200 for Bagging, Random Forest, and Gradient Boosting, showing how quickly each ensemble's error stabilises.

**Task 7 — Hyperparameter Tuning**
Grid Search with 5-fold cross-validation over each model's hyperparameters, followed by a summary table comparing base vs tuned accuracy.

---

## 📁 Key Files

| File | Description |
|------|-------------|
| `Data_Analysis_4th.py` | All seven tasks — base models, depth/estimator studies, tuning |
| `report.pdf` / `.docx` / `.tex` | Methodology, plots, tuning results, and conclusions |
| `summary_results.csv` | Generated base-vs-tuned comparison table |
| `Data_Analysis_2026_3rd_Case_Data.csv` | Input dataset (place alongside the script) |

---

## 🚀 Running

```bash
pip install pandas numpy scikit-learn matplotlib
python Data_Analysis_4th.py
```

Saves two plots (`task5_depth_vs_error.png`, `task6_nestimators_vs_error.png`) and `summary_results.csv`.

---

## 📊 Key Results

| Method | Base Accuracy | Tuned Accuracy |
|--------|---------------|----------------|
| Decision Tree (depth 2) | 0.9396 | 0.9528 |
| Bagging | 0.9622 | 0.9660 |
| **Random Forest** | **0.9641** | **0.9673** |
| Gradient Boosting | 0.9604 | 0.9622 |

**Random Forest** was the best model overall. Tree depth minimised test error around depth 6; ensemble error stabilised after ~50 estimators without overfitting. Tuning chose `max_features = 3` for the Random Forest — heavier feature randomness improved tree decorrelation.

---

*Academic coursework — Aristotle University of Thessaloniki. Not for resubmission in other academic contexts.*
