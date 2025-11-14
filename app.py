# app.py
import streamlit as st

st.set_page_config(
    page_title="Attrition Risk Dashboard",
    layout="wide"
)

st.title("📊 Attrition Risk Dashboard – IBM HR Analytics")

st.write(
    """
    Selamat datang di dashboard *Attrition Risk*.

    Gunakan menu di sebelah kiri (sidebar):

    - **Data Overview**: melihat ringkasan data dan statistik attrition  
    - **Model Performance**: melihat performa model Logistic Regression  
    - **Predict Attrition Risk**: simulasi risiko resign untuk 1 karyawan  
    """
)
