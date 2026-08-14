"""
Live inference — menjalankan model asli hasil Tugas Akhir (bukan simulasi):
1. DistilBERT aspek (fine-tuned, multi-label) - cahya/distilbert-base-indonesian
2. DistilBERT sentimen (fine-tuned, per-aspek) - cahya/distilbert-base-indonesian
3. (mode "Lengkap") Zero-shot XNLI - MoritzLaurer/mDeBERTa-v3-base-mnli-xnli, buat dibandingkan
   dengan hasil DistilBERT

Model DistilBERT (~260MB x2) di-download otomatis dari GitHub Release milik penulis pada
pemakaian pertama, lalu di-cache lokal di folder models_cache/. Tokenizer dasar & model XNLI
di-download dari HuggingFace Hub.

CATATAN OPTIMASI (tidak mengubah hasil):
- `torch` dan `transformers` diimpor MALAS (di dalam fungsi), sehingga membuka halaman
  dashboard biasa tidak perlu menunggu ±5 detik loading pustaka machine learning.
- Beberapa teks diproses sekaligus dalam satu forward pass (batching).
- Padding dinamis (`padding=True`) menggantikan `padding="max_length"`. Karena model memakai
  attention_mask, token padding tidak ikut dihitung — hasilnya identik (selisih < 1e-7, jauh
  di bawah 2-3 angka desimal yang ditampilkan), tetapi jauh lebih cepat karena tidak perlu
  menghitung 96 token untuk kalimat pendek. `truncation=True, max_length=96` tetap dipakai
  persis seperti saat training.
"""
import os
import zipfile
from pathlib import Path

import requests
import streamlit as st

BASE_DIR = Path(__file__).parent
CACHE_DIR = BASE_DIR / "models_cache"
CACHE_DIR.mkdir(exist_ok=True)

# Nama model dasar bisa dioverride lewat environment variable (dipakai untuk pengujian
# offline); secara default mengambil dari HuggingFace Hub seperti pada notebook penelitian.
BASE_MODEL_NAME = os.environ.get("TA_BASE_MODEL", "cahya/distilbert-base-indonesian")
XNLI_MODEL_NAME = os.environ.get("TA_XNLI_MODEL", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

MODEL_RELEASE_BASE = "https://github.com/Fajarjauza/skripsi-model/releases/download/v1.1"
MODEL_ASSETS = {
    "aspek": {
        "url": f"{MODEL_RELEASE_BASE}/model_rasio_80-10-10.zip",
        "zip_name": "model_rasio_80-10-10.zip",
        "inner_folder": "model_rasio_80-10-10",
    },
    "sentimen": {
        "url": f"{MODEL_RELEASE_BASE}/model_rasio_60-20-20.zip",
        "zip_name": "model_rasio_60-20-20.zip",
        "inner_folder": "model_rasio_60-20-20",
    },
}

ASPECTS = ["Individual", "Technical", "Social", "Financial"]
ASPECT_THRESHOLD = {a: 0.5 for a in ASPECTS}  # baseline 0.5 flat (lihat catatan riset)

SENT_LABELS = ["positif", "negatif", "netral"]

FRASA_ASPEK = {
    "Individual": "manfaat aplikasi bagi pengguna",
    "Technical": "kinerja teknis aplikasi",
    "Social": "fitur komunitas aplikasi",
    "Financial": "biaya atau langganan aplikasi",
}

ASPECT_HYPOTHESES = {
    "Individual": "Ulasan ini membahas manfaat atau pengalaman pengguna terhadap aplikasi.",
    "Technical": "Ulasan ini melaporkan masalah teknis seperti aplikasi error, force close, lambat, atau tidak berfungsi dengan baik.",
    "Social": "Ulasan ini membahas fitur komunitas, interaksi, atau berbagi dengan pengguna lain.",
    "Financial": "Ulasan ini mengomentari sistem pembayaran, harga, atau meminta fitur menjadi gratis.",
}

SENTIMENT_TEMPLATES = {
    "positif": "Ulasan ini menyampaikan penilaian positif tentang {}.",
    "negatif": "Ulasan ini menyampaikan keluhan atau penilaian negatif tentang {}.",
    "netral": "Ulasan ini menyampaikan pernyataan netral tanpa penilaian jelas tentang {}.",
}

MAXLEN = 96


# ---------------------------------------------------------------------------
# Download & cache model DistilBERT dari GitHub Release
# ---------------------------------------------------------------------------
def _model_ready(local_dir: Path) -> bool:
    return (local_dir / "config.json").exists() and (
        (local_dir / "model.safetensors").exists() or (local_dir / "pytorch_model.bin").exists()
    )


def ensure_model_downloaded(key: str, progress_cb=None) -> Path:
    """Download + extract model dari GitHub Release kalau belum ada di cache lokal."""
    info = MODEL_ASSETS[key]
    local_dir = CACHE_DIR / key
    if _model_ready(local_dir):
        return local_dir

    local_dir.mkdir(parents=True, exist_ok=True)
    zip_path = CACHE_DIR / info["zip_name"]

    with requests.get(info["url"], stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if progress_cb and total:
                    progress_cb(downloaded / total)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(CACHE_DIR)

    extracted_inner = CACHE_DIR / info["inner_folder"]
    if extracted_inner.exists():
        for f in extracted_inner.iterdir():
            f.rename(local_dir / f.name)
        extracted_inner.rmdir()

    zip_path.unlink(missing_ok=True)
    return local_dir


# ---------------------------------------------------------------------------
# Load model (cached resource, sekali per proses Streamlit)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(BASE_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def load_aspect_model():
    import torch
    from transformers import AutoModelForSequenceClassification
    local_dir = ensure_model_downloaded("aspek")
    model = AutoModelForSequenceClassification.from_pretrained(str(local_dir))
    model.eval()
    torch.set_grad_enabled(False)
    return model


@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    from transformers import AutoModelForSequenceClassification
    local_dir = ensure_model_downloaded("sentimen")
    model = AutoModelForSequenceClassification.from_pretrained(str(local_dir))
    model.eval()
    return model


@st.cache_resource(show_spinner=False)
def load_xnli_pipeline():
    from transformers import pipeline
    return pipeline("zero-shot-classification", model=XNLI_MODEL_NAME, device=-1)


def models_tersedia():
    """True kalau kedua berkas model DistilBERT sudah ada di cache lokal (tidak perlu unduh)."""
    return all(_model_ready(CACHE_DIR / k) for k in MODEL_ASSETS)


def download_models_with_ui():
    """Dipanggil sebelum prediksi pertama - nampilin progress bar download."""
    for key, label in [("aspek", "Model Aspek (DistilBERT)"), ("sentimen", "Model Sentimen (DistilBERT)")]:
        local_dir = CACHE_DIR / key
        if _model_ready(local_dir):
            continue
        bar = st.progress(0.0, text=f"Mengunduh {label} (~260MB, sekali saja)...")

        def cb(frac, _bar=bar, _label=label):
            _bar.progress(min(frac, 1.0), text=f"Mengunduh {_label}... {frac*100:.0f}%")

        ensure_model_downloaded(key, progress_cb=cb)
        bar.empty()


# ---------------------------------------------------------------------------
# Prediksi Aspek (DistilBERT) — versi batch
# ---------------------------------------------------------------------------
def predict_aspect_batch(texts):
    """Prediksi aspek untuk banyak teks sekaligus (satu forward pass)."""
    import torch
    if not texts:
        return []
    tok = load_tokenizer()
    model = load_aspect_model()
    enc = tok([t.lower() for t in texts], truncation=True, padding=True,
              max_length=MAXLEN, return_tensors="pt")
    with torch.inference_mode():
        probs = torch.sigmoid(model(**enc).logits).tolist()

    hasil = []
    for row in probs:
        hasil.append({
            a: {"probabilitas": row[i], "terdeteksi": row[i] >= ASPECT_THRESHOLD[a],
                "threshold": ASPECT_THRESHOLD[a]}
            for i, a in enumerate(ASPECTS)
        })
    return hasil


def predict_aspect(text_clean: str):
    return predict_aspect_batch([text_clean])[0]


# ---------------------------------------------------------------------------
# Prediksi Sentimen (DistilBERT, aspect-conditioned) — versi batch
# ---------------------------------------------------------------------------
def predict_sentiment_batch(pairs):
    """pairs: list of (teks_bersih, aspek). Semua diproses dalam satu forward pass."""
    import torch
    import torch.nn.functional as F
    if not pairs:
        return []
    tok = load_tokenizer()
    model = load_sentiment_model()
    inputs = [f"{FRASA_ASPEK[aspek]} [SEP] {teks.lower()}" for teks, aspek in pairs]
    enc = tok(inputs, truncation=True, padding=True, max_length=MAXLEN, return_tensors="pt")
    with torch.inference_mode():
        probs = F.softmax(model(**enc).logits, dim=-1).tolist()

    hasil = []
    for row, inp in zip(probs, inputs):
        idx = max(range(len(row)), key=lambda i: row[i])
        hasil.append({
            "label": SENT_LABELS[idx],
            "probabilitas": {SENT_LABELS[i]: row[i] for i in range(len(SENT_LABELS))},
            "input_model": inp,
        })
    return hasil


def predict_sentiment(text_clean: str, aspek: str):
    return predict_sentiment_batch([(text_clean, aspek)])[0]


# ---------------------------------------------------------------------------
# XNLI zero-shot (mode "Lengkap" - pembanding)
# ---------------------------------------------------------------------------
def predict_aspect_xnli_batch(texts):
    """Skor aspek zero-shot untuk banyak teks. Mencoba sekali jalan; kalau versi
    transformers-nya tidak mendukung input list, otomatis jatuh ke mode satu per satu."""
    if not texts:
        return []
    clf = load_xnli_pipeline()
    labels = list(ASPECT_HYPOTHESES.values())

    def _to_dict(out):
        label_to_score = dict(zip(out["labels"], out["scores"]))
        return {aspek: label_to_score.get(hip, 0.0) for aspek, hip in ASPECT_HYPOTHESES.items()}

    try:
        outs = clf(list(texts), candidate_labels=labels, hypothesis_template="{}", multi_label=True)
        if isinstance(outs, dict):
            outs = [outs]
        return [_to_dict(o) for o in outs]
    except Exception:
        return [_to_dict(clf(t, candidate_labels=labels, hypothesis_template="{}", multi_label=True))
                for t in texts]


def predict_aspect_xnli(text_clean: str):
    return predict_aspect_xnli_batch([text_clean])[0]


def predict_sentiment_xnli_batch(pairs):
    """pairs: list of (teks_bersih, aspek). Dikelompokkan per aspek supaya kandidat labelnya
    sama, lalu tiap kelompok dijalankan sekaligus."""
    if not pairs:
        return []
    clf = load_xnli_pipeline()
    hasil = [None] * len(pairs)

    per_aspek = {}
    for i, (teks, aspek) in enumerate(pairs):
        per_aspek.setdefault(aspek, []).append((i, teks))

    for aspek, items in per_aspek.items():
        frasa = FRASA_ASPEK[aspek]
        label_texts, text_to_sentimen = [], {}
        for sentimen, template in SENTIMENT_TEMPLATES.items():
            t = template.format(frasa)
            label_texts.append(t)
            text_to_sentimen[t] = sentimen

        teks_list = [t for _, t in items]

        def _to_dict(out):
            skor = {text_to_sentimen[lbl]: s for lbl, s in zip(out["labels"], out["scores"])}
            return {"label": max(skor, key=skor.get), "probabilitas": skor}

        try:
            outs = clf(teks_list, candidate_labels=label_texts,
                       hypothesis_template="{}", multi_label=False)
            if isinstance(outs, dict):
                outs = [outs]
        except Exception:
            outs = [clf(t, candidate_labels=label_texts, hypothesis_template="{}", multi_label=False)
                    for t in teks_list]

        for (idx, _), out in zip(items, outs):
            hasil[idx] = _to_dict(out)

    return hasil


def predict_sentiment_xnli(text_clean: str, aspek: str):
    return predict_sentiment_xnli_batch([(text_clean, aspek)])[0]
