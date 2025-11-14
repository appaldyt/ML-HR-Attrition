# model_training.py
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


@st.cache_resource
def train_model(df: pd.DataFrame):
    """
    Preprocessing + training Logistic Regression.
    Mengembalikan:
      - model terlatih (pipeline)
      - X, y (fitur & target)
      - df_clean (data setelah cleaning)
      - metrics (dict: akurasi, precision, recall, f1, confusion matrix, dll)
    """

    # 1. Drop kolom yang tidak relevan
    cols_to_drop = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
    df_clean = df.drop(columns=cols_to_drop)

    # 2. Encode target
    df_clean["Attrition_flag"] = df_clean["Attrition"].map({"Yes": 1, "No": 0})
    y = df_clean["Attrition_flag"]
    X = df_clean.drop(columns=["Attrition", "Attrition_flag"])

    # 3. Pisahkan numerik & kategorikal
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # 4. Preprocessor
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    # 5. Model Logistic Regression
    log_model = LogisticRegression(max_iter=1000, class_weight="balanced")

    clf = Pipeline(steps=[("preprocess", preprocessor), ("model", log_model)])

    # 6. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 7. Training
    clf.fit(X_train, y_train)

    # 8. Evaluation
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "report": report,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }

    return clf, X, y, df_clean, metrics
