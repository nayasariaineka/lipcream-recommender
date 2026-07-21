"""
Lip Cream Recommender - Streamlit App
Diadaptasi dari notebook: lip_cream_recommender_final_revisi (4).ipynb
Kelompok 8 - Algoritma & Pemrograman II

Versi ini memakai:
- CLIP zero-shot untuk prediksi undertone (tidak perlu training / dataset)
- Color blend untuk simulasi visual shade di bibir (tanpa GPU)

Cara jalankan lokal:
    pip install streamlit torch transformers pillow opencv-python-headless numpy pandas
    streamlit run streamlit_app.py
"""

import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# ----------------------------------------------------------------------------
# Konfigurasi halaman
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Lip Cream Recommender", page_icon="💄", layout="centered")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------
# Load model (di-cache supaya tidak reload tiap interaksi)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model CLIP...")
def load_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor


@st.cache_resource(show_spinner=False)
def load_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# ----------------------------------------------------------------------------
# Database shade lip cream (sama seperti Tahap 5 di notebook)
# ----------------------------------------------------------------------------
@st.cache_data
def load_shade_db():
    data_shade = [
        {"nama_shade": "Terracotta Red", "hex": "#C1440E", "kategori_undertone": "warm",
         "deskripsi": "warm terracotta red lip color with orange undertone"},
        {"nama_shade": "Golden Coral", "hex": "#FF7F50", "kategori_undertone": "warm",
         "deskripsi": "warm golden coral lip color, bright and peachy"},
        {"nama_shade": "Brick Orange", "hex": "#C1440E", "kategori_undertone": "warm",
         "deskripsi": "warm brick orange lip shade for tan skin"},
        {"nama_shade": "Honey Nude", "hex": "#D2A679", "kategori_undertone": "warm",
         "deskripsi": "warm honey nude lip color, natural everyday shade"},
        {"nama_shade": "Rosy Mauve", "hex": "#B76E79", "kategori_undertone": "cool",
         "deskripsi": "cool rosy mauve lip color with pink undertone"},
        {"nama_shade": "Berry Wine", "hex": "#722F37", "kategori_undertone": "cool",
         "deskripsi": "cool deep berry wine lip color, bold and elegant"},
        {"nama_shade": "Cool Pink", "hex": "#E75480", "kategori_undertone": "cool",
         "deskripsi": "cool bright pink lip color with blue undertone"},
        {"nama_shade": "Lavender Plum", "hex": "#8E4585", "kategori_undertone": "cool",
         "deskripsi": "cool lavender plum lip shade, purple toned"},
        {"nama_shade": "Nude Beige", "hex": "#D2B48C", "kategori_undertone": "neutral",
         "deskripsi": "neutral nude beige lip color, balanced tone"},
        {"nama_shade": "Soft Rose", "hex": "#C08081", "kategori_undertone": "neutral",
         "deskripsi": "neutral soft rose lip color, subtle everyday shade"},
        {"nama_shade": "Classic Red", "hex": "#B22222", "kategori_undertone": "neutral",
         "deskripsi": "neutral classic red lip color, universally flattering"},
        {"nama_shade": "Mocha Brown", "hex": "#8B5A2B", "kategori_undertone": "neutral",
         "deskripsi": "neutral mocha brown lip shade, warm-cool balanced"},
    ]
    return pd.DataFrame(data_shade)


# ----------------------------------------------------------------------------
# Fungsi deteksi wajah / bibir (Tahap 7 notebook, + fix bug: cek image None)
# ----------------------------------------------------------------------------
def deteksi_area_bibir(image_path, min_ukuran_wajah_ratio=0.15):
    face_cascade = load_face_cascade()
    image = cv2.imread(image_path)
    if image is None:
        return None, None, False

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w, _ = image.shape

    wajah = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(wajah) == 0:
        return image_rgb, None, False

    x, y, bw, bh = max(wajah, key=lambda box: box[2] * box[3])

    rasio_wajah = (bw * bh) / (w * h)
    if rasio_wajah < min_ukuran_wajah_ratio:
        return image_rgb, None, False

    lx1 = x + int(bw * 0.30)
    lx2 = x + int(bw * 0.70)
    ly1 = y + int(bh * 0.62)
    ly2 = y + int(bh * 0.80)

    lx1, ly1 = max(0, lx1), max(0, ly1)
    lx2, ly2 = min(w, lx2), min(h, ly2)

    if lx2 <= lx1 or ly2 <= ly1:
        return image_rgb, None, False

    return image_rgb, (lx1, ly1, lx2, ly2), True


def simulasi_shade_colorblend(image_path, hex_color, alpha=0.5):
    image_rgb, area_bibir_box, berhasil = deteksi_area_bibir(image_path)
    if not berhasil or image_rgb is None:
        return image_rgb, False

    lx1, ly1, lx2, ly2 = area_bibir_box
    warna_rgb = tuple(int(hex_color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))

    hasil = image_rgb.copy()
    area = hasil[ly1:ly2, lx1:lx2].astype(np.float32)
    warna_layer = np.full_like(area, warna_rgb, dtype=np.float32)
    area_blend = (area * (1 - alpha) + warna_layer * alpha).astype(np.uint8)
    hasil[ly1:ly2, lx1:lx2] = area_blend
    return hasil, True


# ----------------------------------------------------------------------------
# Prediksi undertone (CLIP zero-shot, Tahap 6 notebook)
# ----------------------------------------------------------------------------
def prediksi_undertone_clip(image_path, clip_model, clip_processor):
    image = Image.open(image_path).convert("RGB")
    label_teks = [
        "a photo of a face with warm undertone skin, yellow or golden hue",
        "a photo of a face with cool undertone skin, pink or blue hue",
        "a photo of a face with neutral undertone skin, balanced hue",
    ]
    kategori = ["warm", "cool", "neutral"]

    inputs = clip_processor(text=label_teks, images=image, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        skor = outputs.logits_per_image.softmax(dim=1).cpu().numpy().flatten()

    idx_terbaik = int(np.argmax(skor))
    return kategori[idx_terbaik], dict(zip(kategori, skor.round(3)))


def rekomendasi_shade(image_path, df_shade, clip_model, clip_processor, top_n=3):
    undertone_terdeteksi, skor_undertone = prediksi_undertone_clip(image_path, clip_model, clip_processor)

    kandidat = df_shade[df_shade["kategori_undertone"] == undertone_terdeteksi].copy()

    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(text=list(kandidat["deskripsi"]), images=image, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = clip_model(**inputs)
        skor_shade = outputs.logits_per_image.softmax(dim=1).cpu().numpy().flatten()

    kandidat["skor_kecocokan"] = skor_shade
    hasil = kandidat.sort_values("skor_kecocokan", ascending=False).head(top_n)
    return undertone_terdeteksi, skor_undertone, hasil


# ----------------------------------------------------------------------------
# UI Streamlit
# ----------------------------------------------------------------------------
st.title("💄 Lip Cream Recommender")
st.write("Selamat datang di aplikasi rekomendasi lip cream berdasarkan undertone kulit wajah.")

nama = st.text_input("Masukkan nama")

foto = st.file_uploader("Upload foto wajah (tampak depan, pencahayaan cukup)", type=["jpg", "jpeg", "png"])

if st.button("Tampilkan"):
    if not nama:
        st.warning("Isi nama dulu ya.")
    elif foto is None:
        st.success(f"Halo, {nama}! Aplikasi berhasil berjalan. Silakan upload foto wajah untuk dapat rekomendasi shade.")
    else:
        st.success(f"Halo, {nama}! Memproses foto kamu...")

        # Simpan foto upload ke file sementara (fungsi cv2/PIL butuh path file)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(foto.getvalue())
            temp_path = tmp.name

        with st.spinner("Menganalisis undertone & mencari rekomendasi shade..."):
            clip_model, clip_processor = load_clip()
            df_shade = load_shade_db()
            undertone, skor_undertone, rekomendasi = rekomendasi_shade(
                temp_path, df_shade, clip_model, clip_processor, top_n=3
            )
            hasil_visual, berhasil_visual = simulasi_shade_colorblend(
                temp_path, rekomendasi.iloc[0]["hex"]
            )

        st.subheader(f"Undertone terdeteksi: {undertone.upper()}")
        st.caption(
            f"Skor — warm: {skor_undertone['warm']:.3f} | "
            f"cool: {skor_undertone['cool']:.3f} | "
            f"neutral: {skor_undertone['neutral']:.3f}"
        )

        st.subheader("Rekomendasi Shade")
        for _, row in rekomendasi.iterrows():
            col1, col2 = st.columns([1, 5])
            with col1:
                st.markdown(
                    f"<div style='width:40px;height:40px;border-radius:50%;"
                    f"background-color:{row['hex']};border:1px solid #999;'></div>",
                    unsafe_allow_html=True,
                )
            with col2:
                st.write(f"**{row['nama_shade']}** ({row['hex']}) — skor kecocokan: {row['skor_kecocokan']:.3f}")

        st.subheader("Simulasi Visual di Bibir")
        col_asli, col_hasil = st.columns(2)
        with col_asli:
            st.image(foto, caption="Foto asli", use_container_width=True)
        with col_hasil:
            if berhasil_visual and hasil_visual is not None:
                st.image(hasil_visual, caption=f"Simulasi: {rekomendasi.iloc[0]['nama_shade']}", use_container_width=True)
            else:
                st.warning(
                    "Wajah tidak terdeteksi dengan jelas di foto ini, simulasi visual bibir tidak bisa "
                    "ditampilkan. Coba upload foto wajah frontal dengan pencahayaan cukup, tanpa aksesoris "
                    "yang menutupi wajah."
                )
