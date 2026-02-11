import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model_linear.joblib")

st.title("Prediksi Jumlah Pengunjung Berdasarkan hari,musim,suhu,event,harga")

hari_type = st.selectbox(
    "Tipe Hari",
    ["weekday", "weekend"]
)

musim = st.selectbox(
    "Musim",
    ["hujan", "kemarau"]
)

suhu = st.slider(
    "Suhu Rata-rata",
    15, 40, 28
)

event = st.selectbox(
    "Ada Event Budaya",
    ["ya", "tidak"]
)

harga = st.number_input(
    "Harga Tiket (Ribu)",
    0, 100, 50
)

if st.button("Prediksi"):
    
    data_baru = pd.DataFrame([[
        hari_type,
        musim,
        suhu,
        event,
        harga
    ]], columns=[
        "hari_type",
        "musim",
        "suhu_rata_rata",
        "ada_event_budaya",
        "harga_tiket_ribu"
    ])
    
    hasil = model.predict(data_baru)[0]
   

    st.success(f"Prediksi Jumlah Pengunjung: {hasil:.0f}")
	

