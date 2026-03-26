import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model_linear.joblib")

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Pengunjung",
    page_icon="📊",
    layout="centered"
)

# Header
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>
    📊 Prediksi Jumlah Pengunjung
    </h1>
    <p style='text-align: center;'>
    Gunakan parameter di bawah untuk memprediksi jumlah pengunjung wisata 🎯
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# Form input
with st.container():
    st.subheader("📝 Input Data")

    col1, col2 = st.columns(2)

    with col1:
        hari_type = st.selectbox(
            "📅 Tipe Hari",
            ["weekday", "weekend"]
        )

        musim = st.selectbox(
            "🌦️ Musim",
            ["hujan", "kemarau"]
        )

        suhu = st.slider(
            "🌡️ Suhu Rata-rata (°C)",
            15, 40, 28
        )

    with col2:
        event = st.selectbox(
            "🎉 Ada Event Budaya?",
            ["ya", "tidak"]
        )

        harga = st.number_input(
            "💰 Harga Tiket (Ribu Rupiah)",
            min_value=0,
            max_value=100,
            value=50
        )

st.divider()

# Tombol prediksi
if st.button("🔍 Prediksi Sekarang"):
    
    # Data input
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

    # Prediksi
    hasil = model.predict(data_baru)[0]

    st.success("✅ Prediksi Berhasil!")

    # Output hasil
    st.markdown(
        f"""
        <div style='text-align: center; padding: 20px; border-radius: 10px; background-color: #f0f2f6;'>
            <h2>👥 Estimasi Jumlah Pengunjung</h2>
            <h1 style='color: #4CAF50;'>{hasil:.0f} orang</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Insight tambahan
    if hasil > 500:
        st.info("🔥 Prediksi ramai! Siapkan fasilitas lebih banyak.")
    elif hasil > 200:
        st.info("🙂 Pengunjung cukup stabil.")
    else:
        st.warning("⚠️ Pengunjung relatif sepi.")

st.divider()

# Footer
st.caption("Dibuat dengan ❤️ menggunakan Streamlit")