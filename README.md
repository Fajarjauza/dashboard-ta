# Status TERBARU: Sentimen dipindah 100% ke XNLI, DistilBERT sentimen dilepas dari live prediction

## UPDATE (permintaan eksplisit user, setelah beberapa kali hasil aspek Technical dominan di komentar ambigu)
User: *"gua mau lu ambil best epoch aspek dan sentimen gua yang modeling. sentimen positif negatif
netral hanya xnli saja yang digunakan. distilbert untuk perhitungan score saja. jangan masukkan
NILAI GOLD dari penelitian gua untuk bagian menu Prediksi ini biar pure dari model saja hasilnya.
sesuaikan lagi visualisasi dashboardnya."*

Perubahan yang diimplementasikan (`app.py`, `pipeline.py`, `inference.py`, `README.md` — semua
sudah dikirim ke user via SendUserFile):

1. **Best epoch — sudah terverifikasi TANPA perlu ubah kode/bobot.** Cek notebook
   `05a_modeling_aspek.ipynb` & `05b_modeling_sentimen.ipynb`: `TrainingArguments(...,
   save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model=STOP_ON,
   save_total_limit=1)` + `tr.save_model()` setelah `tr.train()` → checkpoint yang tersimpan di
   `models_final/aspek` & `models_final/sentimen` SUDAH pasti best-epoch (eval_loss terendah).
   Juga diverifikasi via `strings training_args.bin` bahwa checkpoint berasal dari rasio terbaik:
   aspek → `ckpt_80-10-10`, sentimen → `ckpt_60-20-20` (cocok dengan `BEST_RATIO_ASPECT` /
   `BEST_RATIO_SENTIMENT` di `data.py`).
2. **Sentimen sekarang SEPENUHNYA dari XNLI zero-shot, di KEDUA mode** (Cepat maupun Lengkap) —
   bukan lagi dari DistilBERT sentimen. `pipeline.py` step 3 (`analisis_banyak`) memanggil
   `inf.predict_sentiment_xnli_batch(pasangan)` langsung ke `hasil[i]["sentimen"][a]`
   (sebelumnya: `inf.predict_sentiment_batch`, model DistilBERT).
   - Konsekuensi: XNLI (mDeBERTa, ±1-2GB) sekarang WAJIB dimuat untuk SEMUA prediksi, termasuk
     mode "Cepat" — bedanya mode Cepat vs Lengkap sekarang cuma ada/tidaknya Topic Modeling (LDA)
     + simulasi label aspek XNLI tambahan. Mode "Cepat" jadi tidak secepat dulu (dulu murni
     DistilBERT, sekarang tetap butuh XNLI).
   - Model DistilBERT sentimen (`models_final/sentimen`) TIDAK dihapus dari repo — tetap dipakai
     untuk menghitung metrik performa statis Bab IV (F1/precision/recall vs Gold Standard di
     halaman Detail Penelitian) — hanya sudah TIDAK dipanggil lagi di jalur live prediction.
   - `inference.py`: `models_tersedia()` sekarang hanya cek folder cache "aspek" (bukan
     "aspek"+"sentimen"). `download_models_with_ui()` cuma mengunduh checkpoint aspek — checkpoint
     sentimen DistilBERT (~260MB) tidak lagi diunduh runtime sama sekali, hemat bandwidth/waktu.
3. **DistilBERT HANYA menghitung skor/probabilitas aspek** (Individual/Technical/Social/Financial,
   threshold 0.5) — perannya tidak berubah dari sebelumnya, cuma sekarang eksplisit didokumentasikan
   sebagai satu-satunya perannya di live prediction.
4. **Nilai Gold Standard dihapus dari halaman Prediksi Komentar.** Sebelumnya ada info-box
   peringatan "presisi rendah Technical/Financial menurut evaluasi vs Gold Standard 720 ulasan" —
   dihapus total dari hasil prediksi komentar pengguna sendiri, supaya hasil yang ditampilkan
   murni output model, bukan dicampur klaim statistik dari skripsi. Nilai Gold Standard TETAP
   ditampilkan apa adanya di halaman "Detail Penelitian" (itu memang laporan hasil penelitian
   statis, bukan bagian yang diminta "pure model").
   - Statistik korpus lain yang sebelumnya dikutip di kartu NO_ASPECT ("10.072 dari 59.620 ulasan")
     juga disederhanakan jadi kalimat generik tanpa angka skripsi, khusus di halaman Prediksi.
5. **Visualisasi disesuaikan** di `app.py`:
   - Hero chips halaman Prediksi: `["🤖 DistilBERT — Skor Aspek", "💬 XNLI — Sentimen", "🧩 LDA Sub-Topik"]`
     (chip "Zero-shot XNLI" terpisah dihapus karena XNLI sekarang bagian inti, bukan fitur mode
     Lengkap saja).
   - Mode radio: `"⚡ Cepat — Aspek DistilBERT + Sentimen XNLI"` / `"🔬 Lengkap — + Topic Modeling &
     Simulasi Label Aspek XNLI"`.
   - Kartu "how it works" #2/#3 dipisah jadi "Skor Aspek (DistilBERT)" dan "Sentimen per Aspek
     (XNLI)" — sebelumnya digabung.
   - Dihapus: kolom perbandingan 2-kolom "sentimen XNLI vs DistilBERT" (badge sama/beda) di
     bagian hasil per-komentar mode Lengkap — sudah tidak relevan karena sentimen sekarang cuma
     1 sumber. Diganti kartu LDA sub-topik langsung tanpa split kolom.
   - Estimasi waktu proses batch diubah dari `4/1 detik per komentar` (Lengkap/Cepat) jadi
     `6/3 detik` — supaya realistis dengan XNLI yang sekarang jalan di kedua mode.
   - Bagian perbandingan ASPEK "DistilBERT vs Simulasi Label Otomatis (XNLI)" (dari fase
     reframing sebelumnya, lihat bawah) TETAP ADA — itu soal aspek, bukan sentimen, dan sudah
     benar secara metodologis (XNLI = simulasi tahap pelabelan, bukan pembanding produksi setara).
6. **PENTING — belum tentu menyelesaikan keluhan asli user.** Kekhawatiran user yang memicu semua
   ini ("MASA MASUKNYA TEKNIKAL JIR" untuk komentar ambigu/nonsensical) adalah soal **klasifikasi
   ASPEK** (DistilBERT, threshold 0.5, precision rendah utk Technical 0.523 & Financial 0.449 vs
   Gold Standard). Pivot sentimen→XNLI ini TIDAK mengubah cara aspek dideteksi sama sekali — kalau
   user tes ulang dan komentar ambigu masih ke-flag Technical, ituECHO dari masalah aspek yang
   sama, bukan sesuatu yang baru rusak. Perlu di-follow-up eksplisit ke user kalau isu itu muncul
   lagi setelah deploy versi ini, supaya tidak salah kira pivot ini gagal.
- Semua file (`app.py`, `pipeline.py`, `inference.py`, `README.md`) sudah dicompile-check
  (`python3 -m py_compile`) tanpa error, dan sudah dikirim ke user via SendUserFile.

---

# Status sebelumnya: reframing "Pembanding" → "Simulasi Pelabelan Otomatis" (XNLI vs DistilBERT)

User koreksi framing: *"fungsi xnli zeroshot itu untuk melabelkan, lalu distilbert itu model untuk
mencari angka kayak f1 accuracy... masa distilbert di vs kan dengan xnli itu salah."* — XNLI dan
DistilBERT bukan dua model produksi yang bersaing; XNLI dipakai di penelitian HANYA untuk melabeli
59.620 data latih secara otomatis (tahap labeling), lalu DistilBERT di-fine-tune dari label
tersebut dan jadi model klasifikasi final yang diukur F1/precision/recall-nya.

Perubahan saat itu (masih berlaku, kecuali bagian sentimen yang sudah di-pivot lagi di atas):
- Chip & label UI: "Pembanding" → "Simulasi Label Otomatis (XNLI)".
- Card "how it works" dan info-box di bawah chart perbandingan aspek dijelaskan ulang: XNLI =
  simulasi tahap pelabelan (bukan model produksi setara), DistilBERT = model klasifikasi final
  hasil fine-tuning dari label tersebut.
- Diverifikasi (via `training_args.bin` → `strings`) bahwa checkpoint yang dipakai app memang
  rasio terbaik: aspek 80:10:10, sentimen 60:20:20 (match `data.py`).

---

# Status sebelumnya: deployment Streamlit Community Cloud — 3 masalah nyata & solusinya

1. **Build gagal: `gensim==4.3.3` gagal compile di Python 3.14.7.** Tidak ada wheel prebuilt
   gensim untuk Python 3.14 (terlalu baru), fallback build-from-source Cython gagal karena
   C-API CPython/numpy yang dipakai kode gensim sudah berubah/dihapus. **Fix: BUKAN via kode** —
   pilih Python 3.11 secara eksplisit di dialog "Advanced settings" saat deploy Streamlit Cloud
   (pin via `runtime.txt` dikonfirmasi tidak berfungsi saat ini — streamlit/streamlit#15326).
2. **"You do not have access to this app or it does not exist"** saat pilih repo di Streamlit
   Cloud, padahal akun & repo benar. Penyebab: GitHub App milik Streamlit belum diberi akses ke
   repo baru tsb (App discope ke "Only select repositories" sebelum repo dibuat). Fix: user buka
   github.com/settings/installations → Streamlit app → Configure → tambahkan repo (atau ganti ke
   "All repositories").
3. **Tampilan dashboard pucat/washed-out** meski hard refresh. Penyebab: folder `.streamlit/`
   (dotfile-prefixed) tidak ikut ter-upload saat upload manual via GitHub browser karena file
   manager OS (Finder/Explorer) menyembunyikan dotfile secara default — dikonfirmasi lewat
   screenshot listing repo user yang memang tidak ada folder `.streamlit`. Fix: user buat file
   baru langsung di GitHub UI dengan path persis `.streamlit/config.toml`, isi tema warna cyan
   (`primaryColor = "#0e7490"`, dll — lihat riwayat chat untuk isi lengkap).
4. **Optimasi tambahan yang ditemukan & diterapkan:** `requirements.txt` ditambah
   `--extra-index-url https://download.pytorch.org/whl/cpu` sebelum baris `torch` — mencegah pip
   menarik wheel CUDA (~2GB+ dependency NVIDIA yang sama sekali tidak dipakai karena Streamlit
   Cloud tanpa GPU). Ditemukan dengan mempelajari repo deploy teman user
   (github.com/FrenhliX/dashboard-ta-ppd, branch default `master` bukan `main`) — teman user juga
   pakai "simulation mode" (data statis, bukan live inference) sebagai shortcut resource, TAPI itu
   sengaja TIDAK direplikasi di sini karena bertentangan dengan requirement inti proyek ini
   (live inference asli, bukan simulasi).
5. User secara eksplisit minta link gaya Streamlit Community Cloud (`https://dashboard-ta-fahad-
   ppd.streamlit.app/`, contoh dari repo temannya) — sehingga platform pilihan akhir adalah
   Streamlit Community Cloud, BUKAN Hugging Face Spaces (v5, `dashboard_ta_v5_hf_spaces.zip`) yang
   sempat direkomendasikan lebih dulu karena RAM lebih besar (16GB vs ±1GB). Paket HF Spaces tetap
   disimpan sebagai cadangan kalau Streamlit Cloud kehabisan resource untuk Mode Lengkap.

## Struktur aplikasi (sejak v3, masih berlaku)
Sidebar 3 menu: **Dashboard Utama**, **Detail Penelitian** (sub-menu 7 halaman),
**Prediksi Komentar** (mode Cepat/Lengkap × Satu/Batch komentar, maks 500 karakter,
maks 10 komentar, min 3 kata).

## File
- `app.py` — routing + layout + UI prediksi
- `charts.py` — semua chart & tabel statis, di-cache `st.cache_resource`
- `ui.py` — CSS + komponen (hero, kpi, card, navcard, section, info, badge, progress_bar)
- `pipeline.py` — orkestrasi prediksi batch (`analisis_banyak`, `siapkan_model`, `prewarm_async`)
- `preprocessing.py`, `inference.py`, `topic_modeling.py`, `topic_names.py`, `data.py`
- `assets/kamus/*.csv`, `assets/lda/<Aspek>/*` + `{bigram,trigram}_phraser.gensim`
- `.streamlit/config.toml` — tema warna (dotfile, gampang ketinggalan saat upload manual — lihat
  masalah #3 di atas)
- Total ukuran project HANYA ±1,5MB (model weights TIDAK dibundel, diunduh runtime dari
  GitHub Release + HuggingFace Hub).

## Riset perbandingan platform (WebSearch, Agustus 2026)

| Platform | RAM free tier | Cocok? |
|---|---|---|
| Streamlit Community Cloud | ±1GB per app | Dipakai (sesuai preferensi format link user); berisiko untuk Mode Lengkap & sekarang mode Cepat juga (XNLI wajib di semua mode sejak pivot terbaru) |
| Hugging Face Spaces "CPU basic" | 2 vCPU / 16GB RAM / ±50GB disk sementara, gratis | Lebih aman, disiapkan sebagai cadangan (v5) |

## OPTIMASI v4 (percepat pergantian halaman & pemodelan, TANPA mengubah hasil) — masih berlaku

| Aspek | Sebelum | Sesudah |
|---|---|---|
| Buka aplikasi pertama kali | 6,17 s | **1,39 s** |
| Rata-rata pindah halaman (cache panas) | ~0,33 s | **0,119 s** |
| Prediksi pertama (klik tombol) | 9,71 s | **1,41 s** (angka ini dari SEBELUM pivot sentimen→XNLI; sejak XNLI wajib di semua mode, prediksi pertama mode Cepat jadi lebih lambat lagi dari 1,41s ini) |
| Batch 10 komentar (mode Cepat) | 1,14 s | **0,40 s** (2,9x) |

Perubahan: lazy import torch/transformers/gensim/Sastrawi, cache chart (`st.cache_resource`),
prewarm background thread, batching penuh, padding dinamis + `torch.inference_mode()`, cache hasil
analisis (`st.cache_data`), `fileWatcherType = "none"` di config.toml, cache stemming per kata.

### Verifikasi kesetaraan hasil (WAJIB dijaga kalau ada perubahan lagi)
Dibandingkan implementasi lama (per-teks, `padding="max_length"`, `no_grad`) vs baru pada
60 + 40 komentar acak: preprocessing/stemming/aspek/sentimen semua **identik** di level string
tampilan (3/2 desimal); selisih numerik mentah ~1e-8 (floating point). Bandingkan string
terformat (`f"{v:.3f}"`), JANGAN `np.round()` (beda dtype float32/float64 menyesatkan).

## Catatan penting yang masih berlaku
- `requirements.txt` mem-pin `numpy==1.26.4` dan `gensim==4.3.3` — JANGAN diubah tanpa
  men-generate ulang `assets/lda/*.gensim`.
- `requirements.txt` juga punya `--extra-index-url https://download.pytorch.org/whl/cpu` sebelum
  `torch` — JANGAN dihapus (lihat masalah #4 di atas).
- Threshold aspek DistilBERT = 0.5 flat (sesuai keputusan notebook 05a sendiri, threshold-tuning
  grid 0.30-0.85 tidak diadopsi karena tidak memenuhi kriteria perbaikan >0.005 F1 macro).
- `TA_BASE_MODEL` / `TA_XNLI_MODEL` env var bisa dipakai untuk pengujian offline.
- Sandbox Claude tidak bisa akses huggingface.co / github.com via curl langsung (403/blocked);
  WebFetch tool berhasil di jalur berbeda. Mode "Lengkap" (XNLI) tidak pernah dieksekusi
  sungguhan di sandbox Claude karena butuh internet ke huggingface.co, tapi berjalan normal di
  platform hosting (Streamlit Cloud / HF Spaces).
- ASPECTS order `["Individual", "Technical", "Social", "Financial"]` dan SENT_LABELS order
  `["positif", "negatif", "netral"]` di `inference.py` SUDAH diverifikasi cocok persis dengan
  urutan kolom/label saat training (`05a_modeling_aspek.ipynb`, `05b_modeling_sentimen.ipynb`) —
  bukan sumber bug.
- Precision Technical (0.523) & Financial (0.449) vs Gold Standard memang rendah secara riil
  (didokumentasikan di skripsi sendiri, traceable ke AUC XNLI rendah 0.47/0.33 saat eksperimen
  hipotesis) — ini penjelasan asli kenapa komentar ambigu sering ke-flag Technical, BUKAN bug
  index-mapping (sempat dicurigai, sudah diverifikasi salah sebelum disampaikan ke user).
