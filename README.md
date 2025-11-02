# 🎙️ Aplikasi Klasifikasi Suara

Ini adalah aplikasi Streamlit untuk memprediksi ucapan "Buka" atau "Tutup" menggunakan model Machine Learning (Random Forest).

Aplikasi ini bisa memprediksi dari audio yang direkam langsung atau dari file yang di-upload.

## 🚀 Cara Menjalankan

1.  **Clone repository ini:**
    ```bash
    git clone [URL_REPO_ANDA]
    cd [NAMA_FOLDER_REPO]
    ```

2.  **Buat environment Conda & install dependencies:**
    ```bash
    # (Ganti 'voice_app' dengan nama env Anda)
    conda create -n voice_app python=3.11
    conda activate voice_app
    
    # Install library dari requirements
    pip install -r requirements.txt
    
    # Install FFmpeg (penting untuk audiorecorder)
    conda install -c conda-forge ffmpeg
    ```

3.  **Jalankan aplikasi Streamlit:**
    ```bash
    streamlit run app.py
    ```

## 📁 Struktur Proyek

* `app.py`: Kode utama aplikasi Streamlit.
* `1_extract_features.py`: Script untuk ekstraksi fitur MFCC (data input tidak disertakan).
* `2_train_model.py`: Script untuk melatih model klasifikasi.
* `data/voice_features.csv`: Dataset fitur yang sudah diekstrak (200 sampel).
* `models/`: Berisi `model_voice.joblib` dan `scaler_voice.joblib`.
* `requirements.txt`: Daftar library Python yang dibutuhkan.
* `.gitignore`: Mengabaikan folder data mentah (`voice_augmented`).