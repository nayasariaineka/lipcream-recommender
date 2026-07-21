import streamlit as st

st.set_page_config(page_title="Lip Cream Recommender")

st.title("💄 Lip Cream Recommender")
st.write("Selamat datang di aplikasi rekomendasi lip cream.")

nama = st.text_input("Masukkan nama")

if st.button("Tampilkan"):
    st.success(f"Halo, {nama}! Aplikasi berhasil berjalan.")
