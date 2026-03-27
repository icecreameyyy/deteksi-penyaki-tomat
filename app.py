import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time
import os

# === 1. Konfigurasi halaman ===
st.set_page_config(page_title="Deteksi Penyakit Tanaman Tomat", page_icon="🍅", layout="wide")

# === 2. Gaya CSS ===
st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #ffe6e6;
    }
    header[data-testid="stHeader"] {
        background-color: #ff4d4d;
    }
    .result-box {
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        font-size: 18px;
        font-weight: bold;
    }
    .healthy {
        background-color: #e6ffe6;
        color: green;
        border: 2px solid green;
    }
    .disease {
        background-color: #f3f3f3;
        color: #b30000;
        border: 2px solid #b30000;
    }
    .treatment-box {
        background-color: #fffbea;
        border: 2px dashed #ffb84d;
        padding: 12px;
        border-radius: 10px;
        margin-top: 10px;
        font-size: 15px;
        color: #5a4d00;
        text-align: left;
    }
    </style>
""", unsafe_allow_html=True)

# === 3. Muat model terbaik ===
MODEL_PATH = "model_deteksi_tomat_best.h5"
if not os.path.exists(MODEL_PATH):
    st.error(f"Model tidak ditemukan: {MODEL_PATH}. Jalankan dulu script pelatihan (latih_model_tomat.py).")
    st.stop()

model = load_model(MODEL_PATH)

# === 4. Daftar kelas ===
class_names = [
    "antraknosa",
    "bercak_daun",
    "busuk_daun",
    "sehat",
]

# === 5. Daftar penanganan ===
treatments = {
    "antraknosa": [
        "Buang dan bakar bagian tanaman yang terinfeksi.",
        "Gunakan fungisida berbahan aktif tembaga atau azoksistrobin.",
        "Jaga kelembapan lahan agar tidak terlalu tinggi."
    ],
    "bercak_daun": [
        "Pangkas daun yang menunjukkan gejala bercak.",
        "Semprotkan fungisida seperti Dithane M-45 atau Benlate.",
        "Gunakan jarak tanam yang cukup untuk sirkulasi udara."
    ],
    "busuk_daun": [
        "Hindari penyiraman berlebihan.",
        "Gunakan fungisida berbahan aktif mancozeb atau metalaksil.",
        "Rotasi tanaman setiap musim tanam."
    ],
    "sehat": [
        "Tanaman dalam kondisi baik.",
        "Pertahankan perawatan rutin dan pengawasan terhadap gejala awal penyakit."
    ]
}

# === 6. Judul halaman ===
st.markdown(
    """
    <style>
        .judul-responsif {
            font-size: 45px; /* Ukuran di Laptop */
            font-weight: bold;
            line-height: 1.2;
            text-align: left;
        }

        /* Kode khusus untuk layar kecil (HP) */
        @media (max-width: 600px) {
            .judul-responsif {
                font-size: 38px; /* Huruf mengecil sedikit di HP agar pas 2-3 baris */
            }
        }
    </style>
    
    <div class="judul-responsif">
        🍅 Deteksi Penyakit Tanaman Tomat
    </div>
    """, 
    unsafe_allow_html=True
)

st.write("Upload gambar tomat (buah/daun), lalu sistem akan mendeteksi jenis penyakit yang menyerang tanaman tomat serta menampilkan cara penanganannya.")

# === 7. Upload gambar ===
uploaded_file = st.file_uploader("Pilih gambar tomat", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="📸 Gambar yang diunggah", use_container_width=True)

    # === 8. Preprocessing gambar ===
    img_resized = img.resize((150, 150))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype("float32") / 255.0

    # === 9. Prediksi ===
    with st.spinner("🔍 Sedang memproses gambar..."):
        time.sleep(0.8)
        prediction = model.predict(img_array)
        confidence = float(np.max(prediction))
        predicted_class = class_names[int(np.argmax(prediction))]

    # === 10. Filter hasil (confidence) ===
    if confidence < 0.6:
        st.warning("⚠️ Gambar tidak dikenali,Pastikan gambar jelas dan merupakan daun atau buah tomat.")
    else:
        pretty_name = predicted_class.replace("_", " ").title()
        css_class = "healthy" if predicted_class == "sehat" else "disease"

        st.markdown(f"<div class='result-box {css_class}'>Hasil Prediksi: {pretty_name}</div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class='treatment-box'>
                <b>🩺 Rekomendasi Penanganan untuk {pretty_name}:</b><br>
                {"<br>".join([f"• {step}" for step in treatments[predicted_class]])}
        </div>
        """, unsafe_allow_html=True)
