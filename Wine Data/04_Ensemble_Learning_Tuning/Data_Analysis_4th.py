# ============================================================
#  Wine Type Classification — Full Assignment Script
#  AEM = 7137
# ============================================================

# ── IMPORTS ─────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier,
                               GradientBoostingClassifier)
from sklearn.metrics import accuracy_score

# ============================================================
# STEP 1 — LOAD & EXPLORE THE DATA
# ============================================================
AEM = 7137

df = pd.read_csv("Data_Analysis_2026_3rd_Case_Data.csv")

print("=" * 60)
print("STEP 1 — DATA EXPLORATION")
print("=" * 60)
print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nColumn names:\n{df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nBasic statistics:\n{df.describe()}")

# ── Identify target column ───────────────────────────────────
# Assumes the last column is the target; adjust if needed
target_col = df.columns[-1]
print(f"\nTarget column: '{target_col}'")
print(f"Class distribution:\n{df[target_col].value_counts()}")

# ============================================================
# STEP 2 — TRAIN / TEST SPLIT
# ============================================================
print("\n" + "=" * 60)
print("STEP 2 — TRAIN / TEST SPLIT")
print("=" * 60)

X = df.drop(columns=[target_col])
y = df[target_col]

p = X.shape[1]          # total number of features
print(f"\nNumber of features (p): {p}")
print(f"max_features for Random Forest (p//2): {p // 2}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=AEM
)

print(f"\nTraining samples : {X_train.shape[0]}")
print(f"Test samples     : {X_test.shape[0]}")

# ============================================================
# STEP 3 — TASKS 1–4: TRAIN BASE MODELS
# ============================================================
print("\n" + "=" * 60)
print("STEP 3 — BASE MODELS (Tasks 1–4)")
print("=" * 60)

# ── Task 1: Decision Tree (max_depth=2) ─────────────────────
dt = DecisionTreeClassifier(max_depth=2, random_state=AEM)
dt.fit(X_train, y_train)
dt_acc = accuracy_score(y_test, dt.predict(X_test))
print(f"\nTask 1 — Decision Tree (max_depth=2)")
print(f"  Test Accuracy : {dt_acc:.4f}  |  Test Error : {1 - dt_acc:.4f}")

# ── Task 2: Bagging (n_estimators=200) ──────────────────────
bag = BaggingClassifier(n_estimators=200, random_state=AEM)
bag.fit(X_train, y_train)
bag_acc = accuracy_score(y_test, bag.predict(X_test))
print(f"\nTask 2 — Bagging (n_estimators=200)")
print(f"  Test Accuracy : {bag_acc:.4f}  |  Test Error : {1 - bag_acc:.4f}")

# ── Task 3: Random Forest (n_estimators=200, max_features=p//2)
rf = RandomForestClassifier(n_estimators=200, max_features=p // 2,
                             random_state=AEM)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))
print(f"\nTask 3 — Random Forest (n_estimators=200, max_features={p // 2})")
print(f"  Test Accuracy : {rf_acc:.4f}  |  Test Error : {1 - rf_acc:.4f}")

# ── Task 4: Gradient Boosting (n_estimators=200, lr=0.1, depth=1)
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                 max_depth=1, random_state=AEM)
gb.fit(X_train, y_train)
gb_acc = accuracy_score(y_test, gb.predict(X_test))
print(f"\nTask 4 — Gradient Boosting (n_estimators=200, lr=0.1, max_depth=1)")
print(f"  Test Accuracy : {gb_acc:.4f}  |  Test Error : {1 - gb_acc:.4f}")

# ============================================================
# STEP 4 — TASK 5: DEPTH vs TEST ERROR (Decision Tree)
# ============================================================
print("\n" + "=" * 60)
print("STEP 4 — TASK 5: Tree Depth vs Test Error")
print("=" * 60)

depths = list(range(1, 21))          # depth 1 through 20
dt_errors = []

for d in depths:
    model = DecisionTreeClassifier(max_depth=d, random_state=AEM)
    model.fit(X_train, y_train)
    err = 1 - accuracy_score(y_test, model.predict(X_test))
    dt_errors.append(err)

print("\nDepth | Test Error")
print("-" * 20)
for d, e in zip(depths, dt_errors):
    print(f"  {d:2d}  |  {e:.4f}")

plt.figure(figsize=(9, 5))
plt.plot(depths, dt_errors, marker='o', linewidth=2,
         color='steelblue', markersize=6)
plt.axvline(x=2, color='red', linestyle='--', alpha=0.7,
            label='Task 1 depth (=2)')
plt.xlabel("Max Depth", fontsize=12)
plt.ylabel("Test Error (1 − Accuracy)", fontsize=12)
plt.title("Task 5 — Decision Tree: Test Error vs Max Depth", fontsize=13)
plt.xticks(depths)
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("task5_depth_vs_error.png", dpi=150)
plt.show()
print("\nPlot saved as 'task5_depth_vs_error.png'")

# ============================================================
# STEP 5 — TASK 6: n_estimators vs TEST ERROR
#          (Bagging, Random Forest, Gradient Boosting)
# ============================================================
print("\n" + "=" * 60)
print("STEP 5 — TASK 6: n_estimators vs Test Error")
print("=" * 60)

n_values = [1] + list(range(10, 201, 10))   # 1, 10, 20, ..., 200

bag_errors, rf_errors, gb_errors = [], [], []

for n in n_values:
    # Bagging
    m = BaggingClassifier(n_estimators=n, random_state=AEM)
    m.fit(X_train, y_train)
    bag_errors.append(1 - accuracy_score(y_test, m.predict(X_test)))

    # Random Forest
    m = RandomForestClassifier(n_estimators=n, max_features=p // 2,
                                random_state=AEM)
    m.fit(X_train, y_train)
    rf_errors.append(1 - accuracy_score(y_test, m.predict(X_test)))

    # Gradient Boosting
    m = GradientBoostingClassifier(n_estimators=n, learning_rate=0.1,
                                    max_depth=1, random_state=AEM)
    m.fit(X_train, y_train)
    gb_errors.append(1 - accuracy_score(y_test, m.predict(X_test)))

print("\nn_est | Bagging Err | RF Err   | GB Err")
print("-" * 45)
for n, b, r, g in zip(n_values, bag_errors, rf_errors, gb_errors):
    print(f"  {n:3d} |   {b:.4f}    | {r:.4f}  | {g:.4f}")

plt.figure(figsize=(11, 6))
plt.plot(n_values, bag_errors, marker='o', linewidth=2,
         label='Bagging', color='steelblue', markersize=5)
plt.plot(n_values, rf_errors, marker='s', linewidth=2,
         label='Random Forest', color='forestgreen', markersize=5)
plt.plot(n_values, gb_errors, marker='^', linewidth=2,
         label='Gradient Boosting', color='tomato', markersize=5)
plt.xlabel("n_estimators", fontsize=12)
plt.ylabel("Test Error (1 − Accuracy)", fontsize=12)
plt.title("Task 6 — Test Error vs n_estimators", fontsize=13)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("task6_nestimators_vs_error.png", dpi=150)
plt.show()
print("\nPlot saved as 'task6_nestimators_vs_error.png'")

# ============================================================
# STEP 6 — TASK 7: HYPERPARAMETER TUNING (Grid Search + CV)
# ============================================================
print("\n" + "=" * 60)
print("STEP 6 — TASK 7: Hyperparameter Tuning")
print("=" * 60)

CV_FOLDS = 5

# ── 7.1 Decision Tree ────────────────────────────────────────
print("\n[7.1] Tuning Decision Tree ...")
dt_grid = {
    "max_depth"       : [1, 2, 3, 4, 5, 6, 8, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf" : [1, 2, 4],
    "criterion"        : ["gini", "entropy"]
}
dt_gs = GridSearchCV(DecisionTreeClassifier(random_state=AEM),
                     dt_grid, cv=CV_FOLDS, scoring="accuracy", n_jobs=-1)
dt_gs.fit(X_train, y_train)
dt_tuned_acc = accuracy_score(y_test, dt_gs.best_estimator_.predict(X_test))
print(f"  Best params : {dt_gs.best_params_}")
print(f"  CV accuracy : {dt_gs.best_score_:.4f}")
print(f"  Test accuracy (tuned): {dt_tuned_acc:.4f}")

# ── 7.2 Bagging ──────────────────────────────────────────────
print("\n[7.2] Tuning Bagging ...")
bag_grid = {
    "n_estimators"    : [50, 100, 150, 200],
    "max_samples"     : [0.6, 0.8, 1.0],
    "max_features"    : [0.6, 0.8, 1.0]
}
bag_gs = GridSearchCV(BaggingClassifier(random_state=AEM),
                      bag_grid, cv=CV_FOLDS, scoring="accuracy", n_jobs=-1)
bag_gs.fit(X_train, y_train)
bag_tuned_acc = accuracy_score(y_test, bag_gs.best_estimator_.predict(X_test))
print(f"  Best params : {bag_gs.best_params_}")
print(f"  CV accuracy : {bag_gs.best_score_:.4f}")
print(f"  Test accuracy (tuned): {bag_tuned_acc:.4f}")

# ── 7.3 Random Forest ────────────────────────────────────────
print("\n[7.3] Tuning Random Forest ...")
rf_grid = {
    "n_estimators" : [50, 100, 200],
    "max_features" : [p // 4, p // 2, int(p ** 0.5), p],
    "max_depth"    : [None, 5, 10, 20],
    "min_samples_split": [2, 5]
}
rf_gs = GridSearchCV(RandomForestClassifier(random_state=AEM),
                     rf_grid, cv=CV_FOLDS, scoring="accuracy", n_jobs=-1)
rf_gs.fit(X_train, y_train)
rf_tuned_acc = accuracy_score(y_test, rf_gs.best_estimator_.predict(X_test))
print(f"  Best params : {rf_gs.best_params_}")
print(f"  CV accuracy : {rf_gs.best_score_:.4f}")
print(f"  Test accuracy (tuned): {rf_tuned_acc:.4f}")

# ── 7.4 Gradient Boosting ────────────────────────────────────
print("\n[7.4] Tuning Gradient Boosting ...")
gb_grid = {
    "n_estimators"  : [100, 200, 300],
    "learning_rate" : [0.01, 0.05, 0.1, 0.2],
    "max_depth"     : [1, 2, 3],
    "subsample"     : [0.7, 1.0]
}
gb_gs = GridSearchCV(GradientBoostingClassifier(random_state=AEM),
                     gb_grid, cv=CV_FOLDS, scoring="accuracy", n_jobs=-1)
gb_gs.fit(X_train, y_train)
gb_tuned_acc = accuracy_score(y_test, gb_gs.best_estimator_.predict(X_test))
print(f"  Best params : {gb_gs.best_params_}")
print(f"  CV accuracy : {gb_gs.best_score_:.4f}")
print(f"  Test accuracy (tuned): {gb_tuned_acc:.4f}")

# ============================================================
# STEP 7 — SUMMARY TABLE
# ============================================================
print("\n" + "=" * 60)
print("STEP 7 — SUMMARY TABLE")
print("=" * 60)

results = {
    "Method": [
        "Decision Tree",
        "Bagging",
        "Random Forest",
        "Gradient Boosting"
    ],
    "Base Accuracy": [dt_acc, bag_acc, rf_acc, gb_acc],
    "Base Error"   : [1-dt_acc, 1-bag_acc, 1-rf_acc, 1-gb_acc],
    "Tuned Accuracy": [dt_tuned_acc, bag_tuned_acc, rf_tuned_acc, gb_tuned_acc],
    "Tuned Error"   : [1-dt_tuned_acc, 1-bag_tuned_acc,
                       1-rf_tuned_acc, 1-gb_tuned_acc],
    "Improvement"   : [dt_tuned_acc - dt_acc,
                       bag_tuned_acc - bag_acc,
                       rf_tuned_acc - rf_acc,
                       gb_tuned_acc - gb_acc]
}

summary_df = pd.DataFrame(results)
summary_df = summary_df.set_index("Method")

# Format to 4 decimal places for display
pd.set_option("display.float_format", "{:.4f}".format)
print(f"\n{summary_df.to_string()}")

best_method = summary_df["Tuned Accuracy"].idxmax()
best_acc    = summary_df["Tuned Accuracy"].max()
print(f"\n Best method (tuned): {best_method} — Accuracy: {best_acc:.4f}")

# Save summary to CSV
summary_df.to_csv("summary_results.csv")
print("\nSummary saved as 'summary_results.csv'")
print("\n All tasks complete.")