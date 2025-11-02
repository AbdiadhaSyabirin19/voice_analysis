import streamlit as st
import librosa
import numpy as np
import joblib
import sys
import io  # <-- Ini sudah ada
from audiorecorder import audiorecorder  # <-- Ini sudah ada

# --- Konfigurasi Path ---
model_path = "models/model_voice.joblib"
scaler_path = "models/scaler_voice.joblib"
# ------------------------

# 1. Muat Model dan Scaler
@st.cache_resource
def load_model_artifacts():
    """Memuat model dan scaler sekali saja."""
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print("Model dan scaler berhasil dimuat.")
        return model, scaler
    except FileNotFoundError:
        st.error(f"Error: File '{model_path}' atau '{scaler_path}' tidak ditemukan.")
        st.error("Pastikan Anda sudah menjalankan '2_train_model.py' terlebih dahulu.")
        return None, None
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        return None, None

model, scaler = load_model_artifacts()

# 2. Fungsi Ekstraksi Fitur (Tidak perlu diubah)
def extract_mfcc_live(audio_file):
    """Memuat file audio dari uploader/bytes dan mengekstrak rata-rata MFCC."""
    try:
        audio, sr = librosa.load(audio_file, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        return mfccs_mean
    except Exception as e:
        st.error(f"Error saat memproses file audio: {e}")
        return None

# 3. Tampilan Aplikasi Streamlit
st.set_page_config(page_title="Klasifikasi Suara", layout="centered")
st.title("🎙️ Aplikasi Klasifikasi Suara")
st.write("Tekan tombol rekam di bawah, ucapkan 'Buka' atau 'Tutup', lalu tekan tombol rekam lagi untuk berhenti.")

if model is None or scaler is None:
    st.stop()

# 4. Recorder (Sedikit diubah untuk kejelasan nama variabel)
st.write("---")
st.subheader("Rekam Suara Anda:")
# Ubah nama variabel dari audio_bytes menjadi audio_segment
audio_segment = audiorecorder("Klik untuk merekam", "Berhenti merekam")
st.write("---")

# 5. Alur Prediksi (Bagian yang Diperbaiki)
if audio_segment:
    st.info("Audio berhasil direkam. Memproses...")

    # --- PERBAIKAN DIMULAI DI SINI ---
    
    # 1. Buat buffer (penampung) di memori
    wav_bytes_buffer = io.BytesIO()
    
    # 2. Ekspor 'AudioSegment' ke buffer sebagai format 'wav'
    audio_segment.export(wav_bytes_buffer, format="wav")
    
    # 3. Ambil 'bytes' dari buffer untuk st.audio
    wav_bytes = wav_bytes_buffer.getvalue()
    st.audio(wav_bytes, format="audio/wav")
    
    # 4. "Putar ulang" buffer ke awal agar bisa dibaca oleh librosa
    wav_bytes_buffer.seek(0)
    
    # --- PERBAIKAN SELESAI ---

    # Ekstraksi Fitur (sekarang menggunakan buffer)
    with st.spinner("Menganalisis audio..."):
        features = extract_mfcc_live(wav_bytes_buffer)
    
    if features is not None:
        try:
            # Reshape fitur menjadi 2D array
            features_2d = features.reshape(1, -1)
            
            # Scaling fitur
            features_scaled = scaler.transform(features_2d)
            
            # Lakukan Prediksi
            prediction = model.predict(features_scaled)
            prediction_proba = model.predict_proba(features_scaled)
            
            # Ambil probabilitas
            proba_buka = prediction_proba[0][model.classes_.tolist().index('buka')]
            proba_tutup = prediction_proba[0][model.classes_.tolist().index('tutup')]
            
            # Tampilkan Hasil
            st.write("---")
            if prediction[0] == 'buka':
                st.success(f"**Prediksi: BUKA** (Kepercayaan: {proba_buka:.2%})")
            else:
                st.success(f"**Prediksi: TUTUP** (Kepercayaan: {proba_tutup:.2%})")
                
            # Tampilkan detail
            with st.expander("Lihat Detail Probabilitas"):
                st.write(f"Probabilitas 'Buka': {proba_buka:.2%}")
                st.write(f"Probabilitas 'Tutup': {proba_tutup:.2%}")

        except Exception as e:
            st.error(f"Error saat melakukan prediksi: {e}")

else:
    st.info("Silakan rekam suara Anda untuk memulai prediksi.")

# Opsional: Opsi Upload File Manual (Tidak berubah)
st.write("---")
st.subheader("Atau Upload File Manual:")
uploaded_file = st.file_uploader("Pilih file audio...", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file, format=uploaded_file.type)
    with st.spinner("Menganalisis file yang di-upload..."):
        features = extract_mfcc_live(uploaded_file)
    
    if features is not None:
        try:
            features_2d = features.reshape(1, -1)
            features_scaled = scaler.transform(features_2d)
            prediction = model.predict(features_scaled)
            prediction_proba = model.predict_proba(features_scaled)
            proba_buka = prediction_proba[0][model.classes_.tolist().index('buka')]
            proba_tutup = prediction_proba[0][model.classes_.tolist().index('tutup')]
            
            st.write("---")
            if prediction[0] == 'buka':
                st.success(f"**Prediksi (dari file): BUKA** (Kepercayaan: {proba_buka:.2%})")
            else:
                st.success(f"**Prediksi (dari file): TUTUP** (Kepercayaan: {proba_tutup:.2%})")
        
        except Exception as e:
            st.error(f"Error saat prediksi file: {e}")