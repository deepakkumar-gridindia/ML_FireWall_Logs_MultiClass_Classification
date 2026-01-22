import streamlit as st
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
# STREAMLIT CONFIG
# =========================================================
st.set_page_config(page_title="Multi-Class ML Evaluation", layout="wide")
st.title("📊 Multi-Class Classification Model Evaluation")
st.write("Comparison of multiple ML models on a multiclass dataset")

# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data():
    return pd.read_excel("log2_MultiClass_12_65K.xlsx")

df = load_data()

X = df.drop(columns=["Action"])
y = df["Action"]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
class_names = label_encoder.classes_

# =========================================================
# DATASET INFO
# =========================================================
st.subheader("📁 Dataset Information")
st.write("Shape:", df.shape)
st.write("Features:", list(X.columns))
st.write("Target Classes:")

class_map = {i: cls for i, cls in enumerate(class_names)}
st.json(class_map)

# =========================================================
# TRAIN TEST SPLIT
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================================
# FEATURE SCALING
# =========================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================
# MODELS
# =========================================================
models = {
    "Logistic Regression": OneVsRestClassifier(
        LogisticRegression(max_iter=500, solver="lbfgs")
    ),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
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
# TRAIN, PREDICT, EVALUATE
# =========================================================
metrics_table = []
classification_reports = {}
confusion_matrices = {}

st.subheader("⚙️ Model Training & Evaluation")

with st.spinner("Training models... please wait ⏳"):
    for model_name, model in models.items():

        if model_name in ["Naive Bayes", "Random Forest (Ensemble)", "XGBoost (Ensemble)"]:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)

        metrics_table.append({
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC": roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted"),
            "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "MCC": matthews_corrcoef(y_test, y_pred)
        })

        classification_reports[model_name] = pd.DataFrame(
            classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        ).transpose()

        confusion_matrices[model_name] = pd.DataFrame(
            confusion_matrix(y_test, y_pred),
            index=class_names,
            columns=class_names
        )

# =========================================================
# DISPLAY RESULTS
# =========================================================
st.subheader("📌 Final Metrics Comparison")
metrics_df = pd.DataFrame(metrics_table).round(4)
st.dataframe(metrics_df, use_container_width=True)

st.subheader("📄 Classification Reports")
for model_name, report_df in classification_reports.items():
    with st.expander(f"{model_name} – Classification Report"):
        st.dataframe(report_df.round(4))

st.subheader("🔢 Confusion Matrices")
for model_name, cm_df in confusion_matrices.items():
    with st.expander(f"{model_name} – Confusion Matrix"):
        st.dataframe(cm_df)

st.success("✅ Model evaluation completed successfully")
