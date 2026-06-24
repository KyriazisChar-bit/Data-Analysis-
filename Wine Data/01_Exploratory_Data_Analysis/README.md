# 🧹 Assignment 1 — Wine Data Cleaning & Exploratory Analysis

The foundation of the wine-analysis portfolio: transforming a raw, error-ridden CSV into a clean, reproducible dataset ready for modelling, and exploring its structure visually.

**Institution:** Aristotle University of Thessaloniki (School of Mechanical Engineering, AUTh)
**Course:** Data Analysis — 8th Semester
**Author:** Kyriazis Charitopoulos 

---

## ⚙️ Tech Stack

- Python 3
- pandas, NumPy
- Matplotlib

---

## 🏗️ What It Does

**Data definition**
Loads the wine dataset and splits it into training (80%) and test (20%) sets using `sample()` with the student ID (**AEM = 7137**) as the seed, guaranteeing reproducibility.

**Duplicate removal**
Detects and drops duplicate rows (6,708 → 5,593).

**Invalid-value detection**
Scans every column's unique values, normalises typo'd `wine_type` entries (via fuzzy matching), and flags garbage tokens (`-999`, `999`, `9999`, `?`, `NaN`).

**Type coercion & missing values**
Forces numeric columns to floats (non-numerics → `NaN`), counts missing values per column, drops empty/unnamed columns, and keeps rows with at most 3 missing values.

**Imputation**
Fills remaining gaps with the column mean for the corresponding wine type (red / white computed separately).

**Outlier detection & transformation**
Identifies outliers with both the IQR method and the Z-score method (3.5σ threshold), then applies a clamp (clip) transformation.

**Exploratory analysis**
Descriptive statistics for continuous and categorical variables, histograms, boxplots (combined and per wine type), and a correlation heatmap.

---

## 📁 Key Files

| File | Description |
|------|-------------|
| `Data1_EN.py` | Full cleaning + EDA pipeline |
| `report.pdf` / `.docx` / `.tex` | Methodology, code excerpts, results, and commentary |
| `Data_Analysis_2026.csv` | Raw input dataset (place alongside the script) |

---

## 🚀 Running

```bash
pip install pandas numpy matplotlib
python Data1_EN.py
```

---

## 📊 Key Results

| Stage | Rows |
|-------|------|
| Raw sample | 6,708 |
| After deduplication | 5,593 |
| After full cleaning | 5,043 |

Final split by type: **3,931 white**, **1,380 red**. The `pH`, `alcohol`, and `quality` columns revealed clearly invalid entries (e.g. pH > 14, alcohol > 100%), addressed during cleaning.

---

*Academic coursework — Aristotle University of Thessaloniki. Not for resubmission in other academic contexts.*

