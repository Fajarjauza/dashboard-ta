# Dashboard TA — ABSA Ulasan Aplikasi Kesehatan Ibu Hamil

Dashboard Streamlit untuk Tugas Akhir *"Aspect-Based Sentiment Analysis Ulasan
Aplikasi Kesehatan Ibu Hamil Menggunakan Zero-shot Classification dan
DistilBERT"* — Fajar Jauza Maylana (1202220123), S1 Sistem Informasi,
Fakultas Rekayasa Industri, Universitas Telkom, 2026.

Seluruh angka pada dashboard diambil langsung dari isi dokumen skripsi
(`data.py`) — tidak ada data yang dikarang. Selain menampilkan hasil
penelitian, dashboard ini juga bisa menjalankan **prediksi live** memakai
model asli hasil fine-tuning (bukan simulasi) pada halaman
**"Prediksi Komentar"**.

## Menjalankan secara lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

Buka `http://localhost:8501` di browser.

**Wajib koneksi internet** saat pertama kali memakai halaman "Prediksi
Komentar":

- Model DistilBERT (aspek & sentimen, ~260MB masing-masing) diunduh otomatis
  sekali dari [GitHub Release milik penulis](https://github.com/Fajarjauza/skripsi-model/releases/tag/v1.1)
  dan disimpan di folder `models_cache/` (dibuat otomatis, jangan di-commit
  ke git — sudah ada di `.gitignore`).
- Tokenizer dasar (`cahya/distilbert-base-indonesian`) dan model zero-shot
  (`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`, dipakai di mode "Lengkap")
  diunduh otomatis dari HuggingFace Hub oleh library `transformers` dan
  di-cache di `~/.cache/huggingface`.

Setelah unduhan pertama selesai, pemakaian berikutnya jauh lebih cepat karena
sudah tersimpan lokal.

## Deploy

Karena model zero-shot XNLI (mode "Lengkap") berukuran besar (~1-2GB) dan
proses inference-nya cukup berat, **dijalankan secara lokal di komputer
sendiri lebih disarankan** daripada Streamlit Community Cloud (free tier
Streamlit Cloud punya batas resource yang kemungkinan tidak cukup).

Kalau tetap ingin deploy ke Streamlit Community Cloud (halaman hasil
penelitian statis tetap akan berfungsi normal; halaman "Prediksi Komentar"
mode "Lengkap" berisiko lambat/gagal karena keterbatasan resource):

1. Push seluruh folder ini ke sebuah repository GitHub.
2. Buka [share.streamlit.io](https://share.streamlit.io), hubungkan repo
   tersebut, pilih `app.py` sebagai main file, lalu Deploy.

## Struktur file

- `app.py` — routing 3 menu utama, layout tiap halaman, dan UI halaman prediksi.
- `charts.py` — semua chart & tabel statis, dibangun sekali lalu di-cache (`st.cache_resource`).
- `ui.py` — sistem styling (CSS) + komponen reusable: hero, KPI card, card, nav card, badge, info box, popup loading.
- `pipeline.py` — orkestrasi pipeline prediksi untuk satu / banyak komentar (batch), plus batas input.
- `data.py` — seluruh data penelitian (tabel-tabel dari Bab IV & V skripsi) dalam bentuk pandas DataFrame, plus palet warna.
- `preprocessing.py` — text cleaning & normalisasi kamus (replikasi persis notebook `01_data_selection_preprocessing`).
- `inference.py` — download & load model DistilBERT (aspek/sentimen) + XNLI zero-shot, fungsi prediksi.
- `topic_modeling.py` — tokenisasi/stemming/ngram + lookup dominant topic LDA per aspek (replikasi persis notebook `06a_topic_modeling_subaspek`).
- `topic_names.py` — nama 33 sub-topik LDA hasil interpretasi manual penulis.
- `assets/kamus/` — 3 CSV kamus normalisasi (typo/singkatan, huruf berlebih, non-Indonesia).
- `assets/lda/` — dictionary + model gensim LDA per aspek (Individual/Technical/Social/Financial) + bigram/trigram phraser.
- `.streamlit/config.toml` — tema warna Streamlit.
- `requirements.txt` — dependensi Python.

## Struktur menu

Sidebar hanya berisi **3 menu utama**:

1. **Dashboard Utama** — ringkasan angka kunci, tampilan "Sekilas Hasil" yang bisa diganti-ganti
   (distribusi aspek / sentimen / performa model), tahapan KDD, dan kartu navigasi.
2. **Detail Penelitian** — memunculkan sub-menu berisi 7 halaman:
   Data & Preprocessing · Gold Standard & Reliabilitas · Pelabelan Otomatis (XNLI) ·
   Performa Model DistilBERT · Topic Modeling (LDA) · Validasi Ahli · Kesimpulan & Insight.
3. **Prediksi Komentar** — jalankan pipeline lengkap (preprocessing → deteksi aspek →
   sentimen per aspek → topic modeling → pembanding zero-shot) pada komentar sendiri.

## Batas input pada halaman Prediksi Komentar

- Maksimal **500 karakter** per komentar.
- Mode **satu komentar** atau **banyak komentar (batch)** — batch maksimal **10 komentar**,
  ditulis satu komentar per baris.
- Komentar dengan kurang dari **3 kata** setelah preprocessing otomatis dilewati, mengikuti
  aturan filtering yang dipakai di penelitian.
- Hasil batch dilengkapi ringkasan agregat (KPI, grafik, tabel) dan tombol **unduh CSV**.

Nilai batas ini diatur di `pipeline.py` (`MAX_CHAR`, `MAX_BATCH`, `MIN_KATA`).

## Catatan performa

Beberapa optimasi diterapkan **tanpa mengubah hasil prediksi sedikit pun**
(sudah diverifikasi: angka yang ditampilkan dan seluruh label identik dengan
versi sebelumnya):

1. **Pustaka machine learning dimuat malas.** `torch`, `transformers`, `gensim`,
   dan `Sastrawi` baru diimpor saat halaman Prediksi Komentar benar-benar dipakai,
   bukan saat aplikasi dibuka. Waktu buka aplikasi turun dari ±6 detik ke ±1,4 detik.
2. **Chart statis di-cache.** Seluruh grafik dan tabel yang datanya tetap dibangun
   satu kali lalu dipakai ulang, sehingga berpindah halaman tidak membangun ulang
   puluhan figure Plotly (rata-rata ±0,33 detik → ±0,12 detik per perpindahan).
3. **Model dimuat di latar belakang.** Begitu halaman Prediksi Komentar dibuka,
   model mulai dimuat diam-diam sambil pengguna mengetik, sehingga saat tombol
   ditekan model sudah siap (prediksi pertama ±9,7 detik → ±1,4 detik).
4. **Prediksi diproses sekaligus (batch) + padding dinamis.** Semua komentar
   dihitung dalam satu forward pass, dan token padding tidak ikut dihitung karena
   model memakai `attention_mask`. Bobot model, tokenizer, `max_length=96`, dan
   threshold 0,5 tetap sama persis seperti saat training — hanya cara
   menjalankannya yang lebih efisien (2,9–7,4x lebih cepat).
5. **Hasil analisis di-cache.** Menganalisis komentar yang sama untuk kedua kalinya
   langsung tampil tanpa menjalankan ulang model.
6. **Pemantau perubahan file dimatikan** lewat `.streamlit/config.toml`
   (`fileWatcherType = "none"`), karena setiap kali `transformers` selesai dimuat,
   pemantau Streamlit menelusuri seluruh isi paketnya dan memakan waktu beberapa detik.

## Mengubah data

Semua angka statis terpusat di `data.py`. Untuk memperbarui (misalnya
setelah revisi skripsi), cukup ubah nilai di file tersebut — layout `app.py`
akan otomatis mengikuti.
