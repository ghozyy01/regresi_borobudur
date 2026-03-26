import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model_linear.joblib")

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Pengunjung Wisata",
    page_icon="📊",
    layout="wide"
)

# ================== CUSTOM CSS ==================
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .card {
        padding: 20px;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #2E7D32;
    }
    .subtitle {
        text-align: center;
        color: gray;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# ================== HEADER ==================
st.markdown("<div class='title'>📊 Prediksi Pengunjung Wisata</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Analisis berbasis Machine Learning untuk membantu prediksi jumlah pengunjung 🎯</div>", unsafe_allow_html=True)

# ================== SIDEBAR ==================
with st.sidebar:
    st.header("⚙️ Input Parameter")

    hari_type = st.selectbox("📅 Tipe Hari", ["weekday", "weekend"])
    musim = st.selectbox("🌦️ Musim", ["hujan", "kemarau"])
    suhu = st.slider("🌡️ Suhu (°C)", 15, 40, 28)
    event = st.selectbox("🎉 Event Budaya", ["ya", "tidak"])
    harga = st.number_input("💰 Harga Tiket (Ribu)", 0, 100, 50)

    prediksi_btn = st.button("🔍 Prediksi")

# ================== MAIN CONTENT ==================
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📌 Informasi")
    st.markdown("""
    - Model: **Linear Regression**
    - Tujuan: Memprediksi jumlah pengunjung wisata
    - Faktor:
        - Hari
        - Musim
        - Suhu
        - Event
        - Harga tiket
    """)

with col2:
    st.markdown("### 📊 Status")
    st.metric("Suhu Saat Ini", f"{suhu}°C")
    st.metric("Harga Tiket", f"{harga} rb")

# ================== PREDIKSI ==================
if prediksi_btn:

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

    st.markdown("---")
    st.markdown("## 🎯 Hasil Prediksi")

    # CARD HASIL
    st.markdown(f"""
        <div class='card'>
            <h2 style='text-align:center;'>👥 Estimasi Pengunjung</h2>
            <h1 style='text-align:center; color:#4CAF50;'>{hasil:.0f} orang</h1>
        </div>
    """, unsafe_allow_html=True)

    # Insight
    if hasil > 500:
        st.success("🔥 Sangat ramai! Disarankan tambah fasilitas & staff.")
    elif hasil > 200:
        st.info("🙂 Cukup stabil, kondisi normal.")
    else:
        st.warning("⚠️ Sepi, mungkin perlu promosi.")

    # ================== GRAFIK ==================
    st.markdown("### 📈 Simulasi Pengaruh Harga")

    simulasi_harga = list(range(0, 101, 10))
    prediksi_list = []

    for h in simulasi_harga:
        df = pd.DataFrame([[hari_type, musim, suhu, event, h]], columns=data_baru.columns)
        prediksi_list.append(model.predict(df)[0])

    chart_data = pd.DataFrame({
        "Harga": simulasi_harga,
        "Pengunjung": prediksi_list
    })

    st.line_chart(chart_data.set_index("Harga"))

# ================== FOOTER ==================
st.markdown("---")
st.caption("🚀 Project Machine Learning | Dibuat dengan Streamlit")