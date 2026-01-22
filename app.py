# ML_FireWall_Logs_MultiClass_Classification
Firewall Logs ML Intrusion Detection System

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix
)

# =========================================================
# 1. LOAD DATASET
# =========================================================
df = pd.read_excel("log2_MultiClass_12_65K.xlsx")

X = df.drop(columns=["Action"])
y = df["Action"]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

class_names = label_encoder.classes_

print("Dataset Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nClass Mapping:")
for i, cls in enumerate(label_encoder.classes_):
    print(f"{i} -> {cls}")

# =========================================================
# 2. TRAIN-TEST SPLIT
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================================
# 3. FEATURE SCALING
# =========================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================
# 4. MODELS (ALL)
# =========================================================
models = {
    "Logistic Regression": OneVsRestClassifier(
        LogisticRegression(max_iter=500, solver="lbfgs")
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "Naive Bayes": GaussianNB(),

    "Random Forest (Ensemble)": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost (Ensemble)": XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=len(np.unique(y)),
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
}

# =========================================================
# 5. STORAGE OBJECTS
# =========================================================
metrics_table = []
classification_reports = {}
confusion_matrices = {}

# =========================================================
# 6. TRAIN, PREDICT, EVALUATE
# =========================================================
for model_name, model in models.items():

    # ---- Train & Predict ----
    if model_name in ["Naive Bayes"]:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

    elif model_name in ["Random Forest (Ensemble)", "XGBoost"]:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

    else:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)

    # ---- Metrics ----
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    auc = roc_auc_score(
        y_test,
        y_prob,
        multi_class="ovr",
        average="weighted"
    )

    metrics_table.append({
        "ML Model Name": model_name,
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc
    })

    # ---- Classification Report ----
    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )
    classification_reports[model_name] = pd.DataFrame(report).transpose()

    # ---- Confusion Matrix ----
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices[model_name] = pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names
    )

# =========================================================
# 7. DISPLAY FINAL RESULTS
# =========================================================

# ---- Metrics Comparison Table ----
metrics_df = pd.DataFrame(metrics_table)
print("\n================ FINAL METRICS COMPARISON TABLE ================\n")
print(metrics_df.round(4).to_string(index=False))

# ---- Classification Reports ----
for model_name, report_df in classification_reports.items():
    print(f"\n================ CLASSIFICATION REPORT: {model_name} ================\n")
    print(report_df.round(4))

# ---- Confusion Matrices ----
for model_name, cm_df in confusion_matrices.items():
    print(f"\n================ CONFUSION MATRIX: {model_name} ================\n")
    print(cm_df)



