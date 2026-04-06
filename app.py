import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time
import os

# === 1. Konfigurasi halaman ===
st.set_page_config(page_title="Deteksi Penyakit Tanaman Tomat", page_icon="🍅", layout="centered")

# === 2. Gaya CSS ===
st.markdown("""
    <style>
    .upload-wrapper {
        position: relative;
        width: 100%;
        height: 100px;
        margin-top: 10px;
        margin-bottom: 20px;
    }

    .upload-box-visual {
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        border: 2px dashed #ccc;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 25px;
        background-color: #ffffff;
        z-index: 1;
        /* KRUSIAL: Agar klik bisa "tembus" ke uploader asli */
        pointer-events: none; 
    }

    /* CSS lainnya tetap sama seperti sebelumnya */
    .info-left { display: flex; align-items: center; gap: 15px; }
    .cloud-icon { font-size: 35px; color: #888; }
    .text-main { font-weight: bold; font-size: 15px; color: #333; display: block; }
    .text-sub { font-size: 12px; color: #888; }
    .browse-text { font-weight: bold; text-decoration: underline; color: #333; font-size: 14px; }

    [data-testid="stFileUploader"] {
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        z-index: 10; /* Berada di atas visual */
        opacity: 0; 
    }
    
    [data-testid="stFileUploader"] section {
        padding: 0 !important;
        height: 100px !important;
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
st.markdown('<div class="judul">🍅 Deteksi Penyakit Tanaman Tomat</div>', unsafe_allow_html=True)
st.write("Upload gambar tomat (buah/daun) untuk mendeteksi penyakit dan cara penanganannya.")


# === 7. Upload gambar ===
st.write("Pilih gambar tomat")

st.markdown('''
    <div class="upload-wrapper">
        <!-- Lapisan Bawah (Desain) -->
        <div class="upload-box-visual">
            <div class="info-left">
                <div class="cloud-icon">☁️</div>
                <div>
                    <span class="text-main">Drag and drop file here</span>
                    <span class="text-sub">Limit 200MB per file - JPG, JPEG, PNG</span>
                </div>
            </div>
            <div class="browse-text">Browse files</div>
        </div>
''', unsafe_allow_html=True)

# Lapisan Atas (Fungsi Asli - Transparan)
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

st.markdown('</div>', unsafe_allow_html=True)

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
