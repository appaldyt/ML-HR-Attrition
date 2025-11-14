# pages/3_Predict_Attrition_Risk.py
import streamlit as st
import pandas as pd

from load_data import load_data
from model_training import train_model

st.title("🧮 Predict Attrition Risk for One Employee")

df = load_data()
model, X, y, df_clean, metrics = train_model(df)

num_cols = metrics["num_cols"]
cat_cols = metrics["cat_cols"]

# ==== STATISTIK PENTING UNTUK INTERPRETASI ====
median_age = df["Age"].median()
median_income = df["MonthlyIncome"].median()
median_distance = df["DistanceFromHome"].median()
median_years = df["YearsAtCompany"].median()

st.write(
    "Isi profil karyawan di bawah ini untuk melihat **probabilitas risiko resign** "
    "berdasarkan model Logistic Regression."
)

# 👉 NOTE: contoh profil high-risk
with st.expander("💡 Catatan: Contoh profil karyawan yang berisiko resign tinggi", expanded=True):
    st.markdown(
        """
        Berdasarkan pola data historis IBM HR Attrition, model cenderung memberikan
        **probabilitas resign yang tinggi** pada karyawan dengan karakteristik seperti:

        - Usia masih muda (**18–28 tahun**)
        - Masa kerja sangat singkat (**0–2 tahun di perusahaan**)
        - **Gaji bulanan relatif rendah** dibanding rata-rata
        - Sering lembur (**OverTime = Yes**)
        - Sering melakukan perjalanan dinas (**BusinessTravel = Travel_Frequently**)
        - Status pernikahan **Single**
        - Bekerja di **Sales** (misalnya *Sales Representative*)
        - Jarak rumah ke kantor cukup jauh

        Kamu bisa menggunakan tombol di bawah ini untuk mengisi form dengan
        contoh karyawan berisiko tinggi secara otomatis.
        """
    )

st.divider()

# ==== INISIALISASI SESSION STATE UNTUK FORM ====
if "age_input" not in st.session_state:
    st.session_state["age_input"] = int(median_age)
    st.session_state["income_input"] = int(median_income)
    st.session_state["years_input"] = int(median_years)
    st.session_state["distance_input"] = int(median_distance)
    st.session_state["joblevel_input"] = int(df["JobLevel"].median())
    st.session_state["overtime_input"] = df["OverTime"].mode()[0]
    st.session_state["jobrole_input"] = df["JobRole"].mode()[0]
    st.session_state["marital_input"] = df["MaritalStatus"].mode()[0]
    st.session_state["travel_input"] = df["BusinessTravel"].mode()[0]
    st.session_state["dept_input"] = df["Department"].mode()[0]

# ==== TOMBOL: AUTOFILL CONTOH HIGH-RISK ====
if st.button("⚠ Gunakan contoh karyawan berisiko tinggi"):
    st.session_state["age_input"] = 22
    st.session_state["income_input"] = 2000
    st.session_state["years_input"] = 1
    st.session_state["distance_input"] = 25
    st.session_state["joblevel_input"] = 1
    st.session_state["overtime_input"] = "Yes"
    st.session_state["jobrole_input"] = "Sales Representative"
    st.session_state["marital_input"] = "Single"
    st.session_state["travel_input"] = "Travel_Frequently"
    st.session_state["dept_input"] = "Sales"

st.caption("Tombol di atas akan mengisi form dengan profil contoh karyawan berisiko tinggi.")

st.write("")

# ==== FORM PREDIKSI ====
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    # --- Fitur numerik utama ---
    age = col1.slider(
        "Age", 18, 60, key="age_input"
    )
    monthly_income = col1.slider(
        "Monthly Income",
        int(df["MonthlyIncome"].min()),
        int(df["MonthlyIncome"].max()),
        key="income_input",
    )
    years_at_company = col1.slider(
        "Years at Company", 0, int(df["YearsAtCompany"].max()), key="years_input"
    )
    distance_from_home = col1.slider(
        "Distance From Home (km)",
        int(df["DistanceFromHome"].min()),
        int(df["DistanceFromHome"].max()),
        key="distance_input",
    )
    job_level = col1.selectbox(
        "Job Level", sorted(df["JobLevel"].unique()), key="joblevel_input"
    )

    # --- Fitur kategorikal utama ---
    overtime = col2.selectbox(
        "OverTime", sorted(df["OverTime"].unique()), key="overtime_input"
    )
    job_role = col2.selectbox(
        "Job Role", sorted(df["JobRole"].unique()), key="jobrole_input"
    )
    marital_status = col2.selectbox(
        "Marital Status", sorted(df["MaritalStatus"].unique()), key="marital_input"
    )
    business_travel = col2.selectbox(
        "Business Travel", sorted(df["BusinessTravel"].unique()), key="travel_input"
    )
    department = col2.selectbox(
        "Department", sorted(df["Department"].unique()), key="dept_input"
    )

    submitted = st.form_submit_button("Predict Risk")

if submitted:
    # 1 baris dataframe sesuai struktur X
    input_dict = {}

    # isi default: median (numerik) / modus (kategorikal)
    for col in num_cols:
        input_dict[col] = [df_clean[col].median()]
    for col in cat_cols:
        input_dict[col] = [df_clean[col].mode()[0]]

    # overwrite dengan data input user
    input_dict["Age"] = [age]
    input_dict["MonthlyIncome"] = [monthly_income]
    input_dict["YearsAtCompany"] = [years_at_company]
    input_dict["DistanceFromHome"] = [distance_from_home]
    input_dict["JobLevel"] = [job_level]

    input_dict["OverTime"] = [overtime]
    input_dict["JobRole"] = [job_role]
    input_dict["MaritalStatus"] = [marital_status]
    input_dict["BusinessTravel"] = [business_travel]
    input_dict["Department"] = [department]

    input_df = pd.DataFrame(input_dict)

    # Prediksi probabilitas
    proba = model.predict_proba(input_df)[0][1]  # kelas 1 = Resign
    pred = model.predict(input_df)[0]

    st.subheader("Hasil Prediksi")
    st.write(f"**Probabilitas Resign:** `{proba*100:.2f}%`")

    if pred == 1:
        st.error("Model memprediksi: **BERISIKO RESIGN (1)**")
    else:
        st.success("Model memprediksi: **CENDERUNG BERTAHAN (0)**")

    # ==== PENJELASAN FAKTOR RISIKO (RULE-BASED SEDERHANA) ====
    factors_up = []
    factors_down = []

    # Numerik
    if age < median_age:
        factors_up.append("Usia relatif muda dibanding rata-rata karyawan.")
    else:
        factors_down.append("Usia relatif lebih senior (cenderung lebih stabil).")

    if monthly_income < median_income:
        factors_up.append("Gaji bulanan berada di bawah median perusahaan.")
    else:
        factors_down.append("Gaji bulanan berada di atas median perusahaan.")

    if years_at_company <= 2:
        factors_up.append("Masa kerja masih sangat singkat di perusahaan.")
    elif years_at_company > median_years:
        factors_down.append("Masa kerja sudah cukup lama di perusahaan.")

    if distance_from_home > median_distance:
        factors_up.append("Jarak rumah ke kantor relatif jauh.")
    else:
        factors_down.append("Jarak rumah ke kantor relatif dekat.")

    # Kategorikal
    if overtime == "Yes":
        factors_up.append("Sering lembur (OverTime = Yes).")
    else:
        factors_down.append("Tidak sering lembur (OverTime = No).")

    if business_travel == "Travel_Frequently":
        factors_up.append("Sering melakukan perjalanan dinas (Travel_Frequently).")
    elif business_travel == "Non-Travel":
        factors_down.append("Tidak ada perjalanan dinas (Non-Travel).")

    if marital_status == "Single":
        factors_up.append("Status pernikahan Single (lebih fleksibel berpindah kerja).")
    else:
        factors_down.append("Status pernikahan bukan Single (cenderung lebih stabil).")

    if department == "Sales":
        factors_up.append("Bekerja di Department Sales yang historisnya punya attrition lebih tinggi.")
    elif department == "Human Resources":
        factors_up.append("Bekerja di Human Resources yang punya tekanan koordinasi tinggi.")

    if job_role in ["Sales Representative", "Laboratory Technician"]:
        factors_up.append(f"Job Role {job_role} termasuk role dengan turnover relatif tinggi di data historis.")
    elif job_role in ["Manager", "Research Director"]:
        factors_down.append(f"Job Role {job_role} cenderung lebih stabil dibanding role lain.")

    st.subheader("Interpretasi Faktor Risiko (Versi HR)")

    if factors_up:
        st.markdown("**Faktor yang Meningkatkan Risiko Resign:**")
        for f in factors_up:
            st.markdown(f"- {f}")
    else:
        st.markdown("**Tidak ditemukan faktor risiko utama yang kuat dari profil ini.**")

    if factors_down:
        st.markdown("**Faktor yang Menurunkan Risiko Resign / Bersifat Protektif:**")
        for f in factors_down:
            st.markdown(f"- {f}")

    st.caption(
        "Catatan: penjelasan faktor di atas menggunakan kombinasi model statistik dan aturan sederhana "
        "berdasarkan pola umum pada data historis, sehingga hanya bersifat indikatif."
    )
