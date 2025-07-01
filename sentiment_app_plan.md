# Rencana Implementasi Aplikasi Analisis Sentimen Streamlit

## Ringkasan Fitur Aplikasi:

1.  **Halaman Prediksi Sentimen**:
    *   Menerima input teks dari pengguna.
    *   Melakukan seluruh pipeline pra-pemrosesan pada teks input (termasuk stemming, stopword removal, dan tokenization).
    *   Menggunakan vectorizer TF-IDF dan transformer panjang teks yang sudah dilatih untuk ekstraksi fitur.
    *   Memungkinkan pengguna memilih dari enam model yang sudah dilatih sebelumnya (SVM dan XGBoost dengan rasio split 60:40, 70:30, 80:20).
    *   Menampilkan hasil prediksi sentimen.
    *   Pilihan model di UI akan ditampilkan sebagai "SVM Split 80:20", "XGBoost Split 70:30", dan seterusnya.
2.  **Halaman Hasil Evaluasi**:
    *   Menampilkan metrik evaluasi (Akurasi, Presisi, Recall, F1-Score) dan confusion matrix yang diambil langsung dari notebook Jupyter Anda. Data ini akan ditampilkan sebagaimana adanya, tanpa perhitungan ulang.

## Rencana Detail Implementasi:

### Fase 1: Persiapan Pra-implementasi (Mode Arsitek)

*   **Tujuan**: Memastikan semua komponen yang diperlukan (model dan objek pra-pemrosesan) disimpan dan dapat diakses.
*   **Tindakan**:
    1.  **Simpan Objek Pra-pemrosesan**: Memodifikasi notebook Jupyter (`Analisis_Sentimen_pada_Program_Makan_Bergizi_Gratis_Menggunakan_SVM_dan_XGBoost_Edited.ipynb`) atau membuat script Python terpisah untuk menyimpan objek `TfidfVectorizer` (word-level dan char-level) serta `FunctionTransformer` (untuk fitur panjang teks) yang sudah dilatih. Ini penting karena objek-objek ini diperlukan untuk mengubah teks input baru ke dalam format numerik yang dapat diproses oleh model. Akan menggunakan `joblib.dump` untuk menyimpan objek-objek ini setelah mereka dilatih (fit) pada data pelatihan.

### Fase 2: Pengembangan Aplikasi Streamlit (Mode Kode)

*   **Tujuan**: Membangun aplikasi web Streamlit dengan fitur-fitur yang ditentukan.
*   **Tindakan**:
    1.  **Struktur Proyek**: Membuat direktori baru untuk aplikasi Streamlit (misalnya, `streamlit_app/`) dan menempatkan script utama Streamlit (misalnya, `app.py`) di dalamnya. Juga akan dibuat subdirektori `models/` untuk menyimpan model `.pkl` dan objek pra-pemrosesan yang disimpan.
    2.  **Dependensi**: Membuat file `requirements.txt` yang mencantumkan semua pustaka Python yang diperlukan (seperti `streamlit`, `scikit-learn`, `xgboost`, `pandas`, `numpy`, `Sastrawi`, `nltk`, `joblib`, `matplotlib`, `seaborn`, `preprocessor`, `tqdm`, `scipy.stats`).
    3.  **Modul Pra-pemrosesan**: Membuat modul Python terpisah (misalnya, `preprocessing.py`) yang berisi fungsi-fungsi pra-pemrosesan teks (pembersihan, stemming, tokenization, stopword removal) seperti yang didefinisikan dalam notebook Jupyter. Modul ini akan diimpor oleh script utama Streamlit.
    4.  **Logika Pemuatan Model dan Prediksi**:
        *   Dalam `app.py`, memuat keenam model yang sudah dilatih dan objek `TfidfVectorizer` serta `FunctionTransformer` yang sudah disimpan menggunakan `joblib.load`.
        *   Mengimplementasikan fungsi yang menerima teks mentah, menerapkan pipeline pra-pemrosesan, mengubahnya menggunakan vectorizer dan transformer yang dimuat, lalu membuat prediksi menggunakan model yang dipilih.
    5.  **UI Streamlit - Halaman Prediksi Sentimen**:
        *   Membuat halaman utama untuk prediksi sentimen.
        *   Menambahkan widget input teks bagi pengguna untuk memasukkan teks.
        *   Menambahkan dropdown atau radio button untuk pemilihan model, menampilkan opsi seperti "SVM Split 80:20", "XGBoost Split 70:30", dan seterusnya.
        *   Menampilkan hasil prediksi dengan jelas.
    6.  **UI Streamlit - Halaman Hasil Evaluasi**:
        *   Membuat halaman atau bagian terpisah untuk hasil evaluasi.
        *   Menampilkan metrik evaluasi (Akurasi, Presisi, Recall, F1-Score) dan confusion matrix yang diambil langsung dari analisis di notebook Jupyter. Hasil ini akan disajikan menggunakan fungsi tampilan Streamlit yang sesuai (misalnya, `st.dataframe`, `st.metric`, `st.pyplot` untuk plot).
        *   Memastikan presentasi hasil yang jelas dan mudah dibaca.

## Diagram Alir Aplikasi (Mermaid):

```mermaid
graph TD
    A[Pengguna Membuka Aplikasi Streamlit] --> B{Pilih Halaman: Prediksi Sentimen atau Evaluasi}

    B --> C[Halaman Prediksi Sentimen]
    C --> D[Input Teks]
    C --> E[Dropdown Pemilihan Model]
    D -- Input Pengguna --> F[Modul Pra-pemrosesan]
    F --> G[Muat Objek Pra-pemrosesan (TF-IDF Vectorizer, Transformer Panjang Teks)]
    G --> H[Ekstraksi Fitur]
    H --> I{Muat Model Terpilih}
    I --> J[Lakukan Prediksi]
    J --> K[Tampilkan Hasil Prediksi]

    B --> L[Halaman Hasil Evaluasi]
    L --> M[Tampilkan Metrik Evaluasi (Hardcoded)]
    L --> N[Tampilkan Confusion Matrix (Hardcoded)]