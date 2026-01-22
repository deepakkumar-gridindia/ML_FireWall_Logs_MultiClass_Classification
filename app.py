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


Internet Firewall Data — IDS Machine Learning Model
📘 Dataset Description

The Internet Firewall Data is a publicly-available dataset from the UCI Machine Learning Repository (Dataset ID 542). It contains real network traffic records captured from a university’s firewall and is widely used for classification tasks in network security and intrusion detection research.
Instances: 65,532
Features: 12
Task: Multiclass Classification
Class Labels:
allow
deny
drop
reset-both
(These represent the action taken by the firewall on a given traffic session.)

📊 Feature Overview

Each row in the dataset represents one firewall log entry, and the following 12 attributes are included:

Feature	Description
Source Port	Port number initiating the connection
Destination Port	Receiving port number
NAT Source Port	Source port after NAT translation
NAT Destination Port	Destination port after NAT translation
Action	Target label (firewall decision)
Bytes	Total bytes transferred
Bytes Sent	Bytes sent by the source
Bytes Received	Bytes received by the destination
Packets	Total number of packets
Elapsed Time (sec)	Duration of the session
pkts_sent	Packets sent by the source
pkts_received	Packets received by the destination
(Attribute list adapted from the dataset documentation)	

There are no missing values in the dataset, and the class label (Action) is used as the target in supervised learning tasks.

🤖 Project: Intrusion Detection System (IDS)

This repository contains a machine learning-based Intrusion Detection System (IDS) trained on the Internet Firewall Data. The main goal is to automatically classify network traffic records as benign or potentially malicious based on the firewall’s historical actions.

🔧 Included ML Components

✔ Data preprocessing and feature scaling
✔ Handling of class imbalance (if applicable)
✔ Model training and evaluation
✔ Performance metrics (Accuracy, F1-score, Precision, Recall, Confusion Matrix)
✔ Trained model checkpoint and prediction interface

🧠 Algorithms Compared

You can include any of the following (based on what you used):

Logistic Regression

Random Forest Classifier

Support Vector Machine

Gradient Boosting

XGBoost / LightGBM

Tip: You can modify this list depending on what models you actually experimented with.

🚀 Usage

Clone the repository

git clone https://github.com/<your-username>/<your-repo>.git


Install dependencies

pip install -r requirements.txt


Train model

python train.py


Run inference

python predict.py --input sample.csv

📈 Results

Provide a summary of your model performance here:

Model	Accuracy	F1-Score	Precision	Recall
Random Forest	98.5%	0.98	0.99	0.97
SVM	96.2%	0.95	0.96	0.94
…	…	…	…	…

(Replace with your actual results.)

📜 Citation

If you use this dataset or code in published work, please cite:

Internet Firewall Data [Dataset]. (2019). UCI Machine Learning Repository. https://doi.org/10.24432/C5131M
