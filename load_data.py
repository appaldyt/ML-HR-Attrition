# load_data.py
import streamlit as st
import pandas as pd

@st.cache_data
def load_data(path: str = "HR-Employee-Attrition.csv") -> pd.DataFrame:
    """
    Load dataset IBM Attrition dari file CSV.
    Ubah parameter 'path' kalau lokasi file berbeda.
    """
    df = pd.read_csv(path)
    return df
