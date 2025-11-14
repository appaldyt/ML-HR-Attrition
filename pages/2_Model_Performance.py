# pages/2_Model_Performance.py
import streamlit as st
import pandas as pd

from load_data import load_data
from model_training import train_model

st.title("📈 Model Performance – Logistic Regression")

df = load_data()
model, X, y, df_clean, metrics = train_model(df)

# Metrik utama
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{metrics['accuracy']:.2f}")
col2.metric("Precision (Resign)", f"{metrics['precision']:.2f}")
col3.metric("Recall (Resign)", f"{metrics['recall']:.2f}")
col4.metric("F1-Score (Resign)", f"{metrics['f1']:.2f}")

st.subheader("Confusion Matrix")
cm = metrics["confusion_matrix"]
cm_df = pd.DataFrame(
    cm,
    index=["Actual Stay (0)", "Actual Resign (1)"],
    columns=["Predicted Stay (0)", "Predicted Resign (1)"],
)
st.dataframe(cm_df)

st.caption("• Baris = kondisi sebenarnya, Kolom = hasil prediksi model")

st.subheader("Classification Report (Detail)")
report_df = pd.DataFrame(metrics["report"]).T
st.dataframe(report_df.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}"}))
