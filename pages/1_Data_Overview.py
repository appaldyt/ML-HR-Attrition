# pages/1_Data_Overview.py
import streamlit as st
import pandas as pd

from load_data import load_data
from model_training import train_model

st.title("🏠 Data Overview")

df = load_data()
model, X, y, df_clean, metrics = train_model(df)

# KPI kecil
col1, col2, col3 = st.columns(3)

total_emp = len(df)
attrition_rate = (df["Attrition"].value_counts(normalize=True)["Yes"] * 100).round(2)
avg_age = df["Age"].mean().round(1)

col1.metric("Total Employees", total_emp)
col2.metric("Attrition Rate", f"{attrition_rate}%")
col3.metric("Average Age", avg_age)

st.subheader("Sample Data")
st.dataframe(df.head())

st.subheader("Attrition by Department")
dept_attr = df.groupby(["Department", "Attrition"]).size().unstack().fillna(0)
st.dataframe(dept_attr)

st.subheader("Attrition Distribution")
st.bar_chart(df["Attrition"].value_counts())
