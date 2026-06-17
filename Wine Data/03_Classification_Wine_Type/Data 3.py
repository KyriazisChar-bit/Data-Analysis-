import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =========================
# ΒΑΣΙΚΕΣ ΡΥΘΜΙΣΕΙΣ
# =========================
AEM = 7137
# =========================
# ΦΟΡΤΩΣΗ ΔΕΔΟΜΕΝΩΝ
# =========================
df = pd.read_csv("Data_Analysis_2026_3rd_Case_Data.csv")

print("Shape dataset:", df.shape)
print("Columns:", df.columns.tolist())
print("\nΚατανομή wine_type:")
print(df["wine_type"].value_counts())

# =========================
# ΟΡΙΣΜΟΣ TARGET
# =========================
# Δεν χρησιμοποιούμε τη quality ως predictor
target_col = "wine_type"
excluded_cols = ["wine_type", "quality"]

# Κωδικοποίηση target: π.χ. red/white -> 0/1
le = LabelEncoder()
y = le.fit_transform(df[target_col])

print("\nΚλάσεις target:", list(le.classes_))

# Οι 3 μεταβλητές του Μέρους Α
partA_features = ["alcohol", "pH", "chlorides"]

# =========================
# TRAIN / TEST SPLIT
# =========================
# Εδώ κάνουμε split μία φορά για όλο το dataset
train_df, test_df = train_test_split(
    df,
    test_size=0.30,
    random_state=AEM,
    stratify=df[target_col]
)

print("\nTrain shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("\nΚατανομή Train wine_type:")
print(train_df["wine_type"].value_counts())
print("\nΚατανομή Test wine_type:")
print(test_df["wine_type"].value_counts())

# target για train/test
y_train = le.transform(train_df[target_col])
y_test = le.transform(test_df[target_col])

# =========================
# FUNCTION ΑΞΙΟΛΟΓΗΣΗΣ
# =========================
def evaluate_classifier(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="binary", pos_label=1)
    rec = recall_score(y_test, y_pred, average="binary", pos_label=1)
    f1 = f1_score(y_test, y_pred, average="binary", pos_label=1)

    return {
        "confusion_matrix": cm,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }

# =========================
# ΜΟΝΤΕΛΑ ΜΕΡΟΥΣ Α
# =========================
def get_models_partA():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "LDA": LinearDiscriminantAnalysis(),
        "Naive Bayes": GaussianNB()
    }

# =========================
# ΕΚΤΕΛΕΣΗ ΜΕΡΟΥΣ Α
# =========================
results = []

for feature in partA_features:
    print("\n" + "=" * 60)
    print(f"ΜΕΤΑΒΛΗΤΗ: {feature}")
    print("=" * 60)

    # Μόνο η συγκεκριμένη στήλη
    X_train_feature = train_df[[feature]]
    X_test_feature = test_df[[feature]]

    models = get_models_partA()

    for model_name, model in models.items():
        metrics = evaluate_classifier(
            model,
            X_train_feature,
            X_test_feature,
            y_train,
            y_test
        )

        print(f"\n--- {model_name} ---")
        print("Confusion Matrix:")
        print(metrics["confusion_matrix"])
        print(f"Accuracy : {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall   : {metrics['recall']:.4f}")
        print(f"F1-score : {metrics['f1_score']:.4f}")

        results.append({
            "Feature": feature,
            "Method": model_name,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1-score": metrics["f1_score"],
            "TN": metrics["confusion_matrix"][0, 0],
            "FP": metrics["confusion_matrix"][0, 1],
            "FN": metrics["confusion_matrix"][1, 0],
            "TP": metrics["confusion_matrix"][1, 1]
        })

# =========================
# ΤΕΛΙΚΟΣ ΠΙΝΑΚΑΣ
# =========================
results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("ΣΥΓΚΕΝΤΡΩΤΙΚΟΣ ΠΙΝΑΚΑΣ ΜΕΡΟΥΣ Α")
print("=" * 60)
print(results_df.to_string(index=False))

# Προαιρετικά: ταξινόμηση κατά accuracy
print("\n" + "=" * 60)
print("ΤΑΞΙΝΟΜΗΣΗ ΚΑΤΑ ACCURACY")
print("=" * 60)
print(results_df.sort_values(by="Accuracy", ascending=False).to_string(index=False))