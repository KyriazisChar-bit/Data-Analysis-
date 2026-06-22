import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)

# ==========================================
# 1. BASIC SETTINGS
# ==========================================
AEM = 7137
FILE_NAME = "Data_Analysis_2026_3rd_Case_Data.csv"   # change it if needed

# ==========================================
# 2. DATA LOADING
# ==========================================
df = pd.read_csv(FILE_NAME)

print("Shape dataset:", df.shape)
print("Columns:", df.columns.tolist())
print("\nwine_type distribution:")
print(df["wine_type"].value_counts())

# ==========================================
# 3. FEATURES / TARGET DEFINITION
# ==========================================
feature_cols = [col for col in df.columns if col not in ["wine_type", "quality"]]
target_col = "wine_type"

X = df[feature_cols]
y = df[target_col].map({"white": 1, "red": 0})   # white=0, red=1

print("\nFeatures used:")
print(feature_cols)

# ==========================================
# 4. TRAIN / TEST SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=AEM,
    stratify=y
)

print("\nTrain shape:", X_train.shape)
print("Test shape :", X_test.shape)
print("\nTrain target distribution:")
print(y_train.value_counts())
print("\nTest target distribution:")
print(y_test.value_counts())

# ==========================================
# 5. SCALING (where needed)
# ==========================================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 6. EVALUATION FUNCTION
# ==========================================
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name="Model"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # For ROC/AUC we need probabilities or scores
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = None

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1)
    rec = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    if y_score is not None:
        auc = roc_auc_score(y_test, y_score)
        fpr, tpr, _ = roc_curve(y_test, y_score)
    else:
        auc = np.nan
        fpr, tpr = None, None

    return {
        "model_name": model_name,
        "model": model,
        "confusion_matrix": cm,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "auc": auc,
        "fpr": fpr,
        "tpr": tpr
    }

# ==========================================
# 7. K SELECTION FOR KNN WITH 5-FOLD CV
# ==========================================
print("\n" + "=" * 60)
print("SELECTION OF BEST K FOR KNN")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=AEM)

param_grid = {"n_neighbors": list(range(1, 26))}
knn_grid = GridSearchCV(
    KNeighborsClassifier(),
    param_grid=param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

knn_grid.fit(X_train_scaled, y_train)

best_k = knn_grid.best_params_["n_neighbors"]
best_knn_cv_score = knn_grid.best_score_

print(f"Best K: {best_k}")
print(f"Best CV Accuracy: {best_knn_cv_score:.4f}")

# ==========================================
# 8. MODEL DEFINITION
# ==========================================
models = {
    "KNN": {
        "model": KNeighborsClassifier(n_neighbors=best_k),
        "X_train": X_train_scaled,
        "X_test": X_test_scaled
    },
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000, random_state=AEM),
        "X_train": X_train_scaled,
        "X_test": X_test_scaled
    },
    "LDA": {
        "model": LinearDiscriminantAnalysis(),
        "X_train": X_train,
        "X_test": X_test
    },
    "QDA": {
        "model": QuadraticDiscriminantAnalysis(reg_param=0.1),
        "X_train": X_train,
        "X_test": X_test
    },
    "Naive Bayes": {
        "model": GaussianNB(),
        "X_train": X_train,
        "X_test": X_test
    }
}

# ==========================================
# 9. TRAIN + EVALUATE ALL MODELS
# ==========================================
results = []

print("\n" + "=" * 60)
print("PART B RESULTS")
print("=" * 60)

for model_name, model_info in models.items():
    res = evaluate_model(
        model=model_info["model"],
        X_train=model_info["X_train"],
        X_test=model_info["X_test"],
        y_train=y_train,
        y_test=y_test,
        model_name=model_name
    )

    results.append(res)

    print(f"\n--- {model_name} ---")
    print("Confusion Matrix:")
    print(res["confusion_matrix"])
    print(f"Accuracy : {res['accuracy']:.4f}")
    print(f"Precision: {res['precision']:.4f}")
    print(f"Recall   : {res['recall']:.4f}")
    print(f"F1-score : {res['f1_score']:.4f}")
    print(f"AUC      : {res['auc']:.4f}")

# ==========================================
# 10. SUMMARY TABLE
# ==========================================
summary_table = pd.DataFrame([
    {
        "Method": res["model_name"],
        "Accuracy": res["accuracy"],
        "Precision": res["precision"],
        "Recall": res["recall"],
        "F1-score": res["f1_score"],
        "AUC": res["auc"],
        "TN": res["confusion_matrix"][0, 0],
        "FP": res["confusion_matrix"][0, 1],
        "FN": res["confusion_matrix"][1, 0],
        "TP": res["confusion_matrix"][1, 1]
    }
    for res in results
])

print("\n" + "=" * 60)
print("PART B SUMMARY TABLE")
print("=" * 60)
print(summary_table.to_string(index=False))

print("\n" + "=" * 60)
print("SORT BY AUC")
print("=" * 60)
print(summary_table.sort_values(by="AUC", ascending=False).to_string(index=False))

# ==========================================
# 11. ROC CURVES
# ==========================================
plt.figure(figsize=(8, 6))

for res in results:
    if res["fpr"] is not None and res["tpr"] is not None:
        plt.plot(res["fpr"], res["tpr"], label=f"{res['model_name']} (AUC = {res['auc']:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Part B")
plt.legend()
plt.grid(True)
plt.show()

for res in results:
    if res["fpr"] is not None and res["tpr"] is not None:
        plt.figure(figsize=(8, 6))

        plt.plot(
            res["fpr"],
            res["tpr"],
            linewidth=2,
            label=f"{res['model_name']} (AUC = {res['auc']:.3f})"
        )

        plt.plot([0, 1], [0, 1], linestyle="--")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {res['model_name']}")
        plt.legend()
        plt.grid(True)
        plt.show()
