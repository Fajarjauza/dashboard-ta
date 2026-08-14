"""
Seluruh data pada file ini diekstrak langsung dari dokumen Tugas Akhir:
"Aspect-Based Sentiment Analysis Ulasan Aplikasi Kesehatan Ibu Hamil
Menggunakan Zero-shot Classification dan DistilBERT" - Fajar Jauza Maylana
(1202220123), Program Studi S1 Sistem Informasi, Fakultas Rekayasa Industri,
Universitas Telkom, 2026.
"""

import pandas as pd

# ---------------------------------------------------------------------------
# Palet warna (mengikuti validated categorical palette)
# ---------------------------------------------------------------------------
COLOR = {
    "blue": "#0e7490",  # cyan accent (was blue #2a78d6)
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
    "green": "#008300",
    "violet": "#4a3aa7",
    "red": "#e34948",
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
    "surface": "#fcfcfb",
    "page": "#f9f9f7",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
}

ASPECT_COLOR = {
    "Individual": COLOR["blue"],
    "Technical": COLOR["orange"],
    "Social": COLOR["aqua"],
    "Financial": COLOR["yellow"],
}

SENTIMENT_COLOR = {
    "Positif": COLOR["good"],
    "Netral": COLOR["warning"],
    "Negatif": COLOR["critical"],
}

# ---------------------------------------------------------------------------
# Profil penelitian
# ---------------------------------------------------------------------------
META = {
    "judul": "Aspect-Based Sentiment Analysis Ulasan Aplikasi Kesehatan Ibu "
             "Hamil Menggunakan Zero-shot Classification dan DistilBERT",
    "penulis": "Fajar Jauza Maylana",
    "nim": "1202220123",
    "prodi": "S1 Sistem Informasi",
    "fakultas": "Fakultas Rekayasa Industri",
    "universitas": "Universitas Telkom",
    "tahun": "2026",
    "pembimbing1": "Rahmat Fauzi, S.T., M.T.",
    "pembimbing2": "Riska Yanu Fa'rifah, S.Si., M.Si.",
    "abstrak": (
        "Angka Kematian Ibu (AKI) yang masih tinggi di Indonesia mendorong "
        "pemanfaatan aplikasi kesehatan ibu hamil (mHealth), yang ulasan "
        "penggunanya menyimpan informasi berharga namun belum tergali optimal "
        "oleh analisis sentimen konvensional karena tidak mengidentifikasi "
        "aspek spesifiknya. Penelitian ini menerapkan Zero-shot Classification "
        "berbasis XNLI untuk melabeli aspek dan sentimen secara otomatis, "
        "mengevaluasi performa DistilBERT dalam mengklasifikasikannya, serta "
        "menganalisis pengaruh variasi rasio pembagian data (80:10:10, "
        "70:15:15, 60:20:20) menggunakan kerangka Knowledge Discovery in "
        "Databases (KDD)."
    ),
}

KDD_STAGES = [
    ("1. Data Selection", "Web scraping 100.109 ulasan dari 10 aplikasi mHealth di Google Play Store, lalu data preparation menjadi 68.659 ulasan."),
    ("2. Data Preprocessing", "Unicode normalization, case folding, remove URL/email/HTML, normalize separator, remove special character, whitespace normalization, dictionary normalization, filtering panjang teks -> 59.620 ulasan."),
    ("3. Data Transformation", "Pembuatan Gold standard (500 sampel), automatic labeling aspek & sentimen dengan Zero-shot Classification (XNLI)."),
    ("4. Data Mining", "Fine-tuning DistilBERT untuk klasifikasi aspek & sentimen pada 3 skenario rasio split, serta Topic Modeling (LDA) per aspek."),
    ("5. Interpretation / Evaluation", "Evaluasi Confusion matrix, Accuracy, Precision, Recall, F1-Score terhadap Gold standard, evaluasi Coherence score, dan validasi ahli."),
    ("6. Knowledge", "Insight kebutuhan & keluhan pengguna aplikasi kesehatan ibu hamil per aspek, sebagai bahan pertimbangan pengembangan produk."),
]

# ---------------------------------------------------------------------------
# IV.1 Data Selection - Pengumpulan Data
# ---------------------------------------------------------------------------
df_apps = pd.DataFrame([
    ("Asianparent: Kehamilan & Bayi", 54747, 54.69),
    ("Pregnancy+ (Kehamilan+ I Aplikasi Pelacak)", 21357, 21.33),
    ("Teman Bumil - Kehamilan & Anak", 10218, 10.21),
    ("Hallobumil", 6378, 6.37),
    ("BukuBumil: Aplikasi Ibu Hamil", 5448, 5.44),
    ("Diary Bunda Aplikasi Kehamilan", 1754, 1.75),
    ("BabyCenter Pregnancy Tracker", 148, 0.15),
    ("Ovia Pregnancy Tracker", 39, 0.04),
    ("What to Expect", 14, 0.01),
    ("Glow Pregnancy Tracker", 6, 0.01),
], columns=["Aplikasi", "Jumlah Ulasan", "Persentase"])

FUNNEL_PREP = [
    ("Hasil Web Scraping", 100109),
    ("Setelah Hapus Duplikat Komentar per App", 100109 - 30629),
    ("Setelah Hapus Simbol / No-text Only", 100109 - 30629 - 821),
]
# = 68.659 (Data Preparation)

FUNNEL_FILTER = [
    ("Sebelum Penyaringan", 68659),
    ("Setelah Hapus Ulasan Kosong", 68659),
    ("Setelah Hapus Ulasan < 3 Kata", 59620),
]

PREP_STEPS = [
    "Unicode Normalization", "Case folding", "Remove URL", "Remove Email",
    "Remove HTML Tags", "Normalize Separator", "Remove Special Character",
    "Whitespace Normalization", "Dictionary Normalization", "Data Filtering",
]

# ---------------------------------------------------------------------------
# IV.3.1 Gold standard
# ---------------------------------------------------------------------------
df_sampling = pd.DataFrame([
    ("Random murni", 330),
    ("Kandidat Financial (kata kunci: bayar, harga, premium, langganan, dll.)", 100),
    ("Kandidat Social (kata kunci: komunitas, teman, forum, diskusi, dll.)", 70),
], columns=["Kelompok Sampling", "Jumlah"])

df_gold_aspect_support = pd.DataFrame([
    ("Individual", 357), ("Technical", 185), ("Social", 84), ("Financial", 94),
], columns=["Aspek", "Support (label=1)"])

df_gold_pairs = pd.DataFrame([
    ("Individual", 100), ("Technical", 100), ("Social", 84), ("Financial", 94),
], columns=["Aspek", "Jumlah Pasangan Aspek-Sentimen"])

df_gold_sentiment = pd.DataFrame([
    ("Positif", 292), ("Negatif", 56), ("Netral", 30),
], columns=["Kelas", "Jumlah"])

df_kappa_aspect = pd.DataFrame([
    ("Individual", 0.388, "Fair"),
    ("Technical", 0.634, "Substantial"),
    ("Social", 0.839, "Almost Perfect"),
    ("Financial", 0.892, "Almost Perfect"),
    ("Rata-rata", 0.688, "Substantial"),
], columns=["Aspek", "Nilai Kappa", "Interpretasi"])

KAPPA_SENTIMENT = {"nilai": 0.729, "interpretasi": "Substantial"}

KAPPA_SCALE = pd.DataFrame([
    ("< 0,00", "Poor (tidak ada kesepakatan)"),
    ("0,00 - 0,20", "Slight"),
    ("0,21 - 0,40", "Fair"),
    ("0,41 - 0,60", "Moderate"),
    ("0,61 - 0,80", "Substantial"),
    ("0,81 - 1,00", "Almost Perfect"),
], columns=["Rentang Nilai Kappa", "Interpretasi"])

# ---------------------------------------------------------------------------
# IV.3.2 Aspect Labeling (XNLI) - threshold optimization
# ---------------------------------------------------------------------------
df_auc_scheme = pd.DataFrame([
    ("Individual", 0.642, 0.692),
    ("Technical", 0.458, 0.470),
    ("Social", 0.851, 0.886),
    ("Financial", 0.349, 0.329),
    ("Rata-rata", 0.575, 0.594),
], columns=["Aspek", "AUC Skema A (Bawaan)", "AUC Skema B (Custom)"])

df_threshold_aspect = pd.DataFrame([
    ("Individual", 0.31, 0.7515),
    ("Technical", 0.39, 0.4537),
    ("Social", 0.89, 0.7821),
    ("Financial", 0.72, 0.4348),
], columns=["Aspek", "Threshold Optimal", "F1 (saat pencarian)"])

# distribusi jumlah aspek aktif per ulasan
df_active_aspects = pd.DataFrame([
    ("0 (NO_ASPECT)", 10072), ("1", 38073), ("2", 9814), ("3", 1597), ("4", 64),
], columns=["Jumlah Aspek Aktif", "Jumlah Ulasan"])
TOTAL_ULASAN_PREPROCESSED = 59620

df_aspect_distribution = pd.DataFrame([
    ("Individual", 41685, 84.13),
    ("Technical", 13922, 28.10),
    ("Social", 4162, 8.40),
    ("Financial", 2979, 6.01),
], columns=["Aspek", "Jumlah Ulasan", "% dari Terlabel"])

df_gold_match_aspect = pd.DataFrame([
    ("Individual", 0.31, 0.7515, 0.2167, 357),
    ("Technical", 0.39, 0.4537, 0.2166, 185),
    ("Social", 0.89, 0.7821, 0.7420, 84),
    ("Financial", 0.72, 0.4348, 0.3299, 94),
], columns=["Aspek", "Threshold", "F1-Score", "Kappa#2", "Support (Gold)"])
ASPECT_LABELING_SUMMARY = {"f1_macro": 0.6055, "f1_weighted": 0.6372, "kappa2_avg": 0.3763}

# ---------------------------------------------------------------------------
# IV.3.3 Sentiment Labeling (XNLI)
# ---------------------------------------------------------------------------
df_sentiment_distribution = pd.DataFrame([
    ("Positif", 51519, 82.10), ("Negatif", 10147, 16.17), ("Netral", 1082, 1.72),
], columns=["Kelas Sentimen", "Jumlah Pasangan", "Persentase"])
TOTAL_PASANGAN_ASPEK_SENTIMEN = 62748

df_aspect_sentiment_cross = pd.DataFrame([
    ("Individual", 2386, 5.7, 42, 0.1, 39257, 94.2, 41685),
    ("Technical", 6008, 43.2, 618, 4.4, 7296, 52.4, 13922),
    ("Social", 97, 2.3, 3, 0.1, 4062, 97.6, 4162),
    ("Financial", 1656, 55.6, 419, 14.1, 904, 30.3, 2979),
], columns=["Aspek", "Negatif", "Negatif %", "Netral", "Netral %", "Positif", "Positif %", "Total"])

SENTIMENT_GOLD_MATCH = {
    "end_to_end": {"n": 202, "f1_macro": 0.6344, "f1_weighted": 0.8149, "kappa": 0.6541},
    "sentimen_murni": {"n": 378, "f1_macro": 0.6094, "f1_weighted": 0.8400, "kappa": 0.5880},
}

df_class_report_sentimen_murni_xnli = pd.DataFrame([
    ("Negatif", 0.82, 0.80, 0.81, 56),
    ("Netral", 0.17, 0.07, 0.10, 30),
    ("Positif", 0.89, 0.95, 0.92, 292),
    ("Accuracy", None, None, 0.86, 378),
    ("Macro avg", 0.63, 0.61, 0.61, 378),
    ("Weighted avg", 0.82, 0.86, 0.84, 378),
], columns=["Kelas", "Precision", "Recall", "F1-Score", "Support"])

df_class_report_sentimen_e2e_xnli = pd.DataFrame([
    ("Negatif", 0.89, 0.80, 0.85, 51),
    ("Netral", 0.33, 0.10, 0.15, 20),
    ("Positif", 0.85, 0.97, 0.90, 131),
    ("Accuracy", None, None, 0.84, 202),
    ("Macro avg", 0.69, 0.62, 0.63, 202),
    ("Weighted avg", 0.81, 0.84, 0.81, 202),
], columns=["Kelas", "Precision", "Recall", "F1-Score", "Support"])

df_sentiment_per_aspect_gold = pd.DataFrame([
    ("Individual", 100, 0.4990, 0.8062, 0.3509),
    ("Technical", 100, 0.6604, 0.8599, 0.7163),
    ("Social", 84, 0.4540, 0.8847, 0.2812),
    ("Financial", 94, 0.6081, 0.8083, 0.5516),
], columns=["Aspek", "n", "F1 Macro", "F1 Weighted", "Kappa"])

# ---------------------------------------------------------------------------
# IV.4 Data Mining - training
# ---------------------------------------------------------------------------
TRAIN_PARAMS = pd.DataFrame([
    ("Learning rate", "5×10⁻⁶"),
    ("Label smoothing", "0,1"),
    ("Ukuran batch", "16"),
    ("Panjang token maksimum", "96"),
    ("Maksimum epoch", "12"),
    ("Patience (Early stopping)", "2 epoch"),
    ("Weight decay", "0,01"),
], columns=["Parameter", "Nilai"])

df_ratio_aspect = pd.DataFrame([
    ("60:20:20", 0.9643, 0.9438, 0.8333, 0.6484, 0.7025, 0.7925, 7),
    ("70:15:15", 0.9600, 0.9466, 0.8368, 0.6426, 0.6988, 0.7940, 6),
    ("80:10:10", 0.9668, 0.9495, 0.8502, 0.6553, 0.7098, 0.8005, 7),
], columns=["Rasio", "Akurasi Train", "Akurasi Val", "F1 Macro (Test/XNLI)",
            "F1 Macro (Gold)", "F1 Weighted (Gold)", "Akurasi (Gold)", "Epoch Berhenti"])

df_ratio_sentiment = pd.DataFrame([
    ("60:20:20", 0.9457, 0.9299, 0.7069, 0.6518, 0.8631, 0.8836, 5),
    ("70:15:15", 0.9610, 0.9369, 0.7447, 0.6178, 0.8551, 0.8810, 6),
    ("80:10:10", 0.9627, 0.9361, 0.7405, 0.6373, 0.8599, 0.8836, 6),
], columns=["Rasio", "Akurasi Train", "Akurasi Val", "F1 Macro (Test/XNLI)",
            "F1 Macro (Gold)", "F1 Weighted (Gold)", "Akurasi (Gold)", "Epoch Berhenti"])

BEST_RATIO_ASPECT = "80:10:10"
BEST_RATIO_SENTIMENT = "60:20:20"

# Confusion matrix eval (test/XNLI subset) - model aspek rasio terbaik
df_aspect_test_report = pd.DataFrame([
    ("Individual", 0.975, 0.952, 0.964, None),
    ("Technical", 0.862, 0.915, 0.888, None),
    ("Social", 0.773, 0.913, 0.837, None),
    ("Financial", 0.645, 0.796, 0.712, None),
], columns=["Aspek", "Precision", "Recall", "F1-Score", "Support"])
F1_MACRO_ASPECT_TEST = 0.85

# Classification report model aspek vs GOLD (rasio 80:10:10)
df_aspect_gold_report = pd.DataFrame([
    ("Individual", 0.829, 0.924, 0.874, 357),
    ("Technical", 0.523, 0.438, 0.476, 185),
    ("Social", 0.826, 0.905, 0.864, 84),
    ("Financial", 0.449, 0.372, 0.407, 94),
    ("Macro avg", 0.657, 0.660, 0.655, 720),
    ("Weighted avg", 0.700, 0.725, 0.710, 720),
], columns=["Aspek", "Precision", "Recall", "F1-Score", "Support"])

df_threshold_tuning = pd.DataFrame([
    ("Individual", 0.48, 0.8742, 0.8745, 0.0003),
    ("Technical", 0.64, 0.4765, 0.4724, -0.0041),
    ("Social", 0.80, 0.8636, 0.8625, -0.0011),
    ("Financial", 0.48, 0.4070, 0.4138, 0.0068),
], columns=["Aspek", "Threshold Baru", "F1 Baseline (0,5)", "F1 Tuned", "Delta"])
THRESHOLD_TUNING_MACRO = {"baseline": 0.6553, "tuned": 0.6558, "delta": 0.0005}

# Confusion matrix sentimen (test set besar, n=12.431) rasio 60:20:20
df_sentiment_test_report = pd.DataFrame([
    ("Negatif", 0.77, 0.86, 0.81, 2004),
    ("Netral", 0.36, 0.33, 0.35, 214),
    ("Positif", 0.97, 0.95, 0.96, 10213),
    ("Accuracy", None, None, 0.93, 12431),
    ("Macro avg", 0.70, 0.72, 0.71, 12431),
    ("Weighted avg", 0.93, 0.93, 0.93, 12431),
], columns=["Kelas", "Precision", "Recall", "F1-Score", "Support"])

# Classification report sentimen vs GOLD (rasio 60:20:20, n=378)
df_sentiment_gold_report = pd.DataFrame([
    ("Negatif", 0.72, 0.88, 0.79, 56),
    ("Netral", 0.67, 0.13, 0.22, 30),
    ("Positif", 0.92, 0.96, 0.94, 292),
    ("Accuracy", None, None, 0.88, 378),
    ("Macro avg", 0.77, 0.66, 0.65, 378),
    ("Weighted avg", 0.87, 0.88, 0.86, 378),
], columns=["Kelas", "Precision", "Recall", "F1-Score", "Support"])

df_netral_error = pd.DataFrame([
    ("Positif", 18, 60.0), ("Negatif", 8, 26.7), ("Netral (benar)", 4, 13.3),
], columns=["Prediksi", "Jumlah", "Persentase"])

# ---------------------------------------------------------------------------
# IV.4.4 / IV.5.4 Topic Modeling (LDA)
# ---------------------------------------------------------------------------
df_lda_corpus = pd.DataFrame([
    ("Individual", 41685, 29873),
    ("Technical", 13922, 10727),
    ("Social", 4162, 3952),
    ("Financial", 2979, 2777),
], columns=["Aspek", "Ukuran Subset Awal", "Ukuran Korpus Setelah Pra-pemrosesan"])

df_lda_k_range = pd.DataFrame([
    ("Individual", 29873, "2-20"),
    ("Technical", 10727, "2-16"),
    ("Social", 3952, "2-12"),
    ("Financial", 2777, "2-12"),
], columns=["Aspek", "Jumlah Dokumen", "Rentang K yang Diuji"])

df_lda_coherence = pd.DataFrame([
    ("Individual", 6, 0.4636, 3, 0.4410),
    ("Technical", 16, 0.4115, 16, 0.4111),
    ("Social", 2, 0.3696, 2, 0.3518),
    ("Financial", 12, 0.3467, 12, 0.3601),
], columns=["Aspek", "K Hasil Pencarian", "Coherence (K Pencarian)", "K Final", "Coherence (K Final)"])

df_subtopics = pd.DataFrame([
    (1, "Individual", "Manfaat dan Kemudahan Aplikasi bagi Ibu Hamil", 11045, 36.97),
    (2, "Individual", "Pemantauan Tumbuh Kembang Janin dan Bayi", 10792, 36.13),
    (3, "Individual", "Kendala Penggunaan Awal Aplikasi", 8036, 26.90),
    (4, "Technical", "Ulasan Umum Campuran (Apresiasi dan Kendala Koneksi/Unduh)", 926, 8.63),
    (5, "Technical", "Kesulitan Instalasi dan Pendaftaran, Rating Rendah", 563, 5.25),
    (6, "Technical", "Error dan Aplikasi Lambat Setelah Update", 804, 7.50),
    (7, "Technical", "Kendala Bahasa dan Login/Akses Aplikasi", 597, 5.57),
    (8, "Technical", "Apresiasi Fitur Informasi dan Edukasi", 998, 9.30),
    (9, "Technical", "Apresiasi Kelengkapan Fitur dan Keluhan Konten Komunitas", 497, 4.63),
    (10, "Technical", "Kendala Membuka Aplikasi dan Login Gagal", 526, 4.90),
    (11, "Technical", "Ulasan Umum Campuran (Apresiasi Singkat dan Keluhan Teknis Beragam)", 477, 4.45),
    (12, "Technical", "Kesulitan Membuka/Mengunduh Aplikasi", 665, 6.20),
    (13, "Technical", "Kekecewaan Fitur Berbayar dan Aplikasi Keluar Sendiri", 679, 6.33),
    (14, "Technical", "Masalah Loading dan Login", 608, 5.67),
    (15, "Technical", "Apresiasi Informasi dan Pengalaman Berbagi", 639, 5.96),
    (16, "Technical", "Aplikasi Keluar Sendiri dan Gagal Dibuka (Parah)", 801, 7.47),
    (17, "Technical", "Kebingungan Input Tanggal dan Usia Kehamilan", 1018, 9.49),
    (18, "Technical", "Keluhan Upgrade Memaksa Login Ulang dan Macet", 406, 3.78),
    (19, "Technical", "Keluhan Iklan dan Kegagalan Sistem", 523, 4.88),
    (20, "Social", "Berbagi Pengalaman dengan Sesama Ibu Hamil", 1806, 45.70),
    (21, "Social", "Manfaat Komunitas dan Berbagi Informasi", 2146, 54.30),
    (22, "Financial", "Keluhan Biaya Konsultasi dan Fitur Premium vs Gratis", 185, 6.66),
    (23, "Financial", "Pertanyaan dan Kendala Seputar Tanggal Haid/HPHT", 244, 8.79),
    (24, "Financial", "Kendala Login Setelah Update", 204, 7.35),
    (25, "Financial", "Kegagalan Login dan Pembuatan Akun", 281, 10.12),
    (26, "Financial", "Cara Klaim Hadiah/Pembayaran", 229, 8.25),
    (27, "Financial", "Kesulitan Pembelian Fitur Berbayar", 179, 6.45),
    (28, "Financial", "Kebingungan Setelah Update dan Transaksi Gagal", 238, 8.57),
    (29, "Financial", "Kekecewaan Akun dan Fitur Lambat", 251, 9.04),
    (30, "Financial", "Kendala Login dan Verifikasi Akun Berulang", 213, 7.67),
    (31, "Financial", "Pertanyaan Seputar Member VIP dan Fitur Gratis", 241, 8.68),
    (32, "Financial", "Aplikasi Lambat dan Gagal Dibuka Setelah Update", 275, 9.90),
    (33, "Financial", "Kendala Pendaftaran Akun", 237, 8.53),
], columns=["No", "Aspek", "Sub-Topik", "Jumlah Dokumen", "% dari Aspek"])

TOTAL_SUBTOPICS = 33

# ---------------------------------------------------------------------------
# IV.5.5 Validasi Ahli
# ---------------------------------------------------------------------------
df_validators = pd.DataFrame([
    ("Validator 1", "Dhewa Radya Hanggardha Phat Yoga", "AI Apps Technical Consultant Associate",
     "PT. Mitra Integrasi Informatika", 2, "11 Agustus 2026"),
    ("Validator 2", "Satrio Rahman Wicaksono", "Software Development Engineer in Test",
     "Bank SMBC Indonesia", 5, "10 Agustus 2026"),
], columns=["Validator", "Nama", "Profesi/Jabatan", "Institusi", "Pengalaman (tahun)", "Tanggal Validasi"])

df_validation_statements = pd.DataFrame([
    (1, "Analisis aspek Individual, Technical, Social, Financial relevan untuk memahami "
        "kebutuhan dan keluhan pengguna secara terstruktur", 2, 4, 3.0),
    (2, "App Developer/Product Manager relevan sebagai pengguna utama hasil analisis aspek", 4, 2, 3.0),
    (3, "Hasil analisis aspek berpotensi menjadi bahan pertimbangan prioritas perbaikan fitur/layanan", 4, 4, 4.0),
    (4, "Keempat aspek mencakup dimensi utama yang lazim menjadi perhatian tim pengembang aplikasi kesehatan", 3, 3, 3.0),
    (5, "Sub-topik LDA berpotensi memberikan informasi rinci untuk insight pengembangan produk", 3, 3, 3.0),
], columns=["No", "Pernyataan", "Validator 1", "Validator 2", "Rata-rata"])

df_validation_subtopic = pd.DataFrame([
    ("Individual", 3, 3.67, 3.67, 3.67, 3.67, 3.67, 3.67),
    ("Technical", 16, 3.00, 3.56, 3.28, 3.38, 3.44, 3.41),
    ("Social", 2, 4.00, 4.00, 4.00, 4.00, 4.00, 4.00),
    ("Financial", 12, 2.00, 2.83, 2.42, 3.67, 2.75, 3.21),
], columns=["Aspek", "Jumlah Sub-Topik", "Relevansi V1", "Relevansi V2", "Relevansi Rata-Rata",
            "Kejelasan V1", "Kejelasan V2", "Kejelasan Rata-Rata"])

df_validation_yesno = pd.DataFrame([
    (1, "Pengelompokan empat kategori aspek secara keseluruhan sudah sesuai untuk memetakan "
        "kebutuhan dan keluhan pengguna", "Sebagian", "Sebagian"),
    (2, "Label sub-topik secara keseluruhan mencerminkan pola isu yang lazim ditemui pada "
        "pengembangan aplikasi kesehatan sejenis", "Sebagian", "Sebagian"),
    (3, "Terdapat label atau pengelompokan sub-topik yang berpotensi menyesatkan atau kurang actionable",
        "Ya", "Ya"),
], columns=["No", "Pernyataan", "Validator 1", "Validator 2"])

# ---------------------------------------------------------------------------
# Bab V - Ringkasan & Insight
# ---------------------------------------------------------------------------
df_ratio_summary = pd.DataFrame([
    ("Aspek", "80:10:10", 0.7098, "60:20:20", 0.7025, 0.0073, 0.6553, 0.6484, 0.0069),
    ("Sentimen", "60:20:20", 0.8631, "80:10:10", 0.8599, 0.0032, 0.6518, 0.6373, 0.0145),
], columns=["Model", "Rasio Terbaik", "F1 Weighted Terbaik", "Rasio Kedua",
            "F1 Weighted Kedua", "Selisih Weighted", "F1 Macro Terbaik",
            "F1 Macro Kedua", "Selisih Macro"])

df_gap_test_gold = pd.DataFrame([
    ("Individual", 0.964, 0.874, 0.090),
    ("Technical", 0.888, 0.476, 0.412),
    ("Social", 0.837, 0.864, -0.027),
    ("Financial", 0.712, 0.407, 0.305),
], columns=["Aspek", "F1 Data Test", "F1 Gold", "Selisih"])

df_correlation = pd.DataFrame([
    ("Individual", 0.874, 3, 0.4410),
    ("Technical", 0.476, 16, 0.4111),
    ("Social", 0.864, 2, 0.3518),
    ("Financial", 0.407, 12, 0.3601),
], columns=["Aspek", "F1 Gold (Klasifikasi)", "Jumlah Sub-Topik (LDA)", "Coherence Score (K Final)"])

KNOWLEDGE_SUMMARY = [
    ("Pelabelan Aspek (XNLI)", "Data Transformation",
     "F1 macro 0,6055 | F1 weighted 0,6372 | Kappa#2 rata-rata 0,3763"),
    ("Pelabelan Sentimen (XNLI)", "Data Transformation",
     "Sentimen murni (n=378): F1 weighted 0,8400, F1 macro 0,6094 | "
     "End-to-end (n=202): F1 weighted 0,8149, F1 macro 0,6344"),
    ("Reliabilitas Gold standard (Kappa#1)", "Data Transformation",
     "Aspek rata-rata 0,688 (Substantial); Sentimen 0,729 (Substantial)"),
    ("Model Klasifikasi Aspek (DistilBERT)", "Data Mining & Evaluation",
     "Rasio terbaik 80:10:10 -> F1 weighted Gold 0,7098; F1 macro Gold 0,6553"),
    ("Model Klasifikasi Sentimen (DistilBERT)", "Data Mining & Evaluation",
     "Rasio terbaik 60:20:20 -> F1 weighted Gold 0,8631; F1 macro Gold 0,6518"),
    ("Pengaruh Rasio Pembagian Data", "Data Mining & Evaluation",
     "Rasio optimal berbeda antar model: 80:10:10 untuk aspek, 60:20:20 untuk sentimen"),
    ("Topic Modeling (LDA)", "Data Mining & Evaluation",
     "33 sub-topik teridentifikasi pada 4 aspek (3 Individual, 16 Technical, 2 Social, 12 Financial)"),
]

KPI = {
    "total_ulasan_scraped": 100109,
    "total_ulasan_final": 59620,
    "jumlah_aplikasi": 10,
    "jumlah_aspek": 4,
    "jumlah_subtopik": 33,
    "f1_weighted_aspek": 0.7098,
    "f1_weighted_sentimen": 0.8631,
    "kappa_aspek_avg": 0.688,
    "kappa_sentimen": 0.729,
}
