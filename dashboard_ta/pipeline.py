"""
Orkestrasi pipeline prediksi untuk satu / banyak komentar.

Dipakai oleh halaman "Prediksi Komentar" supaya logika inference terpisah dari UI.

PEMBAGIAN PERAN MODEL (sesuai metodologi penelitian):
- DistilBERT (fine-tuned, rasio 80:10:10 — rasio terbaik) HANYA dipakai untuk menghitung
  skor/probabilitas ASPEK (Individual/Technical/Social/Financial).
- Sentimen (positif/negatif/netral) SEPENUHNYA memakai XNLI zero-shot — bukan DistilBERT
  sentimen — konsisten dengan cara data latih penelitian ini dilabeli.
- Karena itu, model XNLI (mDeBERTa, ±1-2GB) sekarang WAJIB dimuat untuk setiap prediksi,
  termasuk mode "Cepat" (bukan cuma mode "Lengkap" seperti sebelumnya).

CATATAN OPTIMASI (tidak mengubah cara model dipanggil, hanya kecepatan):
- Modul berat (`preprocessing`, `inference`, `topic_modeling` → torch/transformers/gensim)
  diimpor MALAS di dalam fungsi. Membuka halaman dashboard biasa jadi jauh lebih cepat karena
  pustaka machine learning tidak ikut dimuat kalau memang tidak dipakai.
- Semua komentar diproses secara BATCH: satu forward pass untuk deteksi aspek seluruh komentar,
  lalu XNLI dipanggil sekaligus per kelompok aspek untuk seluruh pasangan (komentar, aspek).
"""

ASPECTS = ["Individual", "Technical", "Social", "Financial"]

MAX_CHAR = 500
MAX_BATCH = 10
MIN_KATA = 3


def siapkan_model(mode_lengkap=False):
    """Pastikan berkas model sudah terunduh & termuat sebelum analisis dijalankan.
    XNLI selalu dimuat (bukan cuma mode Lengkap) karena sekarang jadi satu-satunya sumber
    sentimen."""
    import inference as inf
    inf.download_models_with_ui()
    inf.load_tokenizer()
    inf.load_aspect_model()
    inf.load_xnli_pipeline()


def prewarm_async():
    """Muat model di latar belakang begitu halaman prediksi dibuka, supaya ketika tombol
    ditekan model sudah siap. Hanya berjalan kalau berkas model DistilBERT sudah ada di cache
    lokal (kalau belum, biarkan proses unduh berjalan di depan mata pengguna dengan progress
    bar). Tidak mengubah hasil apa pun — hanya memindahkan waktu tunggu ke belakang layar."""
    import threading

    def _run():
        try:
            import inference as inf
            if not inf.models_tersedia():
                return
            inf.load_tokenizer()
            inf.load_aspect_model()
            inf.load_xnli_pipeline()
        except Exception:
            pass  # kalau gagal, prediksi normal tetap akan mencoba memuat ulang

    threading.Thread(target=_run, daemon=True).start()


def analisis_banyak(list_teks, mode_lengkap=False, progress_cb=None):
    """
    Jalankan pipeline lengkap untuk satu atau banyak komentar sekaligus.
    progress_cb(pesan) dipanggil tiap tahap supaya UI bisa menampilkan status.
    """
    import preprocessing as prep
    import inference as inf

    def _p(msg):
        if progress_cb:
            progress_cb(msg)

    # ---- 1. Preprocessing semua komentar -----------------------------------
    _p("Membersihkan teks (preprocessing)...")
    hasil = []
    for teks in list_teks:
        trace = prep.preprocess_with_trace(teks)
        hasil.append({
            "teks_asli": trace["asli"],
            "teks_bersih": trace["setelah_normalisasi_kamus"],
            "jumlah_kata": trace["jumlah_kata"],
            "valid": trace["jumlah_kata"] >= MIN_KATA,
            "aspek": None,
            "aspek_terdeteksi": [],
            "sentimen": {},
            "topik": {},
            "xnli_aspek": None,
        })

    idx_valid = [i for i, h in enumerate(hasil) if h["valid"]]
    if not idx_valid:
        return hasil

    teks_valid = [hasil[i]["teks_bersih"] for i in idx_valid]

    # ---- 2. Skor aspek — DistilBERT (satu forward pass untuk semua komentar) -
    _p(f"Menjalankan DistilBERT — skor aspek ({len(teks_valid)} komentar sekaligus)...")
    aspek_semua = inf.predict_aspect_batch(teks_valid)
    for i, aspek_hasil in zip(idx_valid, aspek_semua):
        hasil[i]["aspek"] = aspek_hasil
        hasil[i]["aspek_terdeteksi"] = [a for a in ASPECTS if aspek_hasil[a]["terdeteksi"]]

    # ---- 3. Sentimen per (komentar, aspek) — XNLI zero-shot (satu-satunya sumber) --
    pasangan, peta = [], []
    for i in idx_valid:
        for a in hasil[i]["aspek_terdeteksi"]:
            pasangan.append((hasil[i]["teks_bersih"], a))
            peta.append((i, a))

    if pasangan:
        _p(f"Menjalankan XNLI zero-shot — sentimen ({len(pasangan)} pasangan aspek sekaligus)...")
        sentimen_semua = inf.predict_sentiment_xnli_batch(pasangan)
        for (i, a), s in zip(peta, sentimen_semua):
            hasil[i]["sentimen"][a] = s

    # ---- 4. Mode Lengkap: LDA + simulasi label aspek XNLI (ilustrasi tambahan) --
    if mode_lengkap and pasangan:
        import topic_modeling as tm

        _p("Menentukan sub-topik (LDA)...")
        for i, a in peta:
            hasil[i]["topik"][a] = tm.prediksi_topik(hasil[i]["teks_bersih"], a)

        _p("Menjalankan zero-shot XNLI — simulasi label aspek...")
        idx_beraspek = [i for i in idx_valid if hasil[i]["aspek_terdeteksi"]]
        if idx_beraspek:
            xnli_aspek = inf.predict_aspect_xnli_batch([hasil[i]["teks_bersih"] for i in idx_beraspek])
            for i, x in zip(idx_beraspek, xnli_aspek):
                hasil[i]["xnli_aspek"] = x

    return hasil


def analisis_satu(teks, mode_lengkap=False, progress_cb=None):
    """Pembungkus untuk satu komentar (memakai jalur batch yang sama)."""
    return analisis_banyak([teks], mode_lengkap, progress_cb)[0]


def parse_batch(raw_text):
    """Pecah input multi-komentar (1 komentar per baris) jadi list bersih."""
    baris = [b.strip() for b in raw_text.splitlines()]
    return [b for b in baris if b]


def _potong(teks, n=60):
    return teks[:n] + ("..." if len(teks) > n else "")


def ringkas_untuk_tabel(hasil_list):
    """Ubah list hasil analisis jadi baris-baris ringkas untuk tabel/unduhan."""
    rows = []
    for i, h in enumerate(hasil_list, start=1):
        if not h["valid"]:
            rows.append({"No": i, "Komentar": _potong(h["teks_asli"]),
                         "Aspek Terdeteksi": "— (teks terlalu pendek)",
                         "Sentimen": "—", "Sub-Topik": "—"})
            continue

        if not h["aspek_terdeteksi"]:
            rows.append({"No": i, "Komentar": _potong(h["teks_asli"]),
                         "Aspek Terdeteksi": "NO_ASPECT",
                         "Sentimen": "—", "Sub-Topik": "—"})
            continue

        sentimen_str = ", ".join(
            f"{a}: {h['sentimen'][a]['label']}" for a in h["aspek_terdeteksi"]
        )
        topik_list = []
        for a in h["aspek_terdeteksi"]:
            t = h["topik"].get(a)
            if t:
                topik_list.append(f"{a}: {t['nama_subtopik']}")
        rows.append({
            "No": i,
            "Komentar": _potong(h["teks_asli"]),
            "Aspek Terdeteksi": ", ".join(h["aspek_terdeteksi"]),
            "Sentimen": sentimen_str,
            "Sub-Topik": "; ".join(topik_list) if topik_list else "—",
        })
    return rows
