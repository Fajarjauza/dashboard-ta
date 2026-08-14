"""
Text preprocessing — replikasi PERSIS dari notebook 01_data_selection_preprocessing.ipynb
milik penulis (basic_text_clean + normalisasi kamus 2 pass dengan fallback huruf berulang).
"""
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import pandas as pd

ASSETS_DIR = Path(__file__).parent / "assets"
KAMUS_DIR = ASSETS_DIR / "kamus"


# ---------------------------------------------------------------------------
# Basic text clean (persis notebook 01)
# ---------------------------------------------------------------------------
def normalize_space(t):
    return re.sub(r"\s+", " ", t).strip()


def basic_text_clean(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = text.replace("&amp;", " dan ").replace("&nbsp;", " ")
    for ch in ["/", "\\", "_", "-"]:
        text = text.replace(ch, " ")
    text = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s]", " ", text)
    return normalize_space(text)


# ---------------------------------------------------------------------------
# Kamus normalisasi (typo/singkatan, huruf berlebih, non-Indonesia)
# ---------------------------------------------------------------------------
def _load_dictionary_csv(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]
    kolom_asli = "kata_asli" if "kata_asli" in df.columns else df.columns[0]
    kolom_benar = "kata_benar" if "kata_benar" in df.columns else df.columns[1]
    mapping = {}
    for _, row in df.iterrows():
        k = str(row[kolom_asli]).strip().lower()
        v = str(row[kolom_benar]).strip().lower()
        if k and k != "nan":
            mapping[k] = v
    return mapping


@lru_cache(maxsize=1)
def _build_combined_mapping():
    """Dimuat MALAS (hanya saat preprocessing pertama kali dipakai) dan hanya sekali.

    Pola regex tiap frasa langsung dikompilasi di sini supaya tidak dikompilasi ulang
    setiap kali fungsi dipanggil — urutan & perilaku penggantiannya tetap sama persis
    seperti kode notebook aslinya, jadi hasilnya tidak berubah.
    """
    typo_mapping = _load_dictionary_csv(KAMUS_DIR / "kamus_typo_singkatan.csv")
    repeated_mapping = _load_dictionary_csv(KAMUS_DIR / "kamus_huruf_berlebih.csv")
    non_indonesia_mapping = _load_dictionary_csv(KAMUS_DIR / "kamus_non_indonesia.csv")
    combined = {}
    combined.update(typo_mapping)
    combined.update(repeated_mapping)
    combined.update(non_indonesia_mapping)
    phrase_mapping = {k: v for k, v in combined.items() if " " in k}
    token_mapping = {k: v for k, v in combined.items() if " " not in k}
    phrase_sorted = sorted(phrase_mapping.items(), key=lambda x: len(x[0]), reverse=True)
    phrase_compiled = [(re.compile(r"\b" + re.escape(s) + r"\b"), tg) for s, tg in phrase_sorted]
    return token_mapping, phrase_compiled


def normalize_repeated_characters_fallback(t):
    return re.sub(r"(.)\1{2,}", r"\1\1", t) if t else ""


def apply_phrase_mapping(t):
    if not t:
        return t
    _, phrase_compiled = _build_combined_mapping()
    for pola, tg in phrase_compiled:
        t = pola.sub(tg, t)
    return normalize_space(t)


def apply_token_mapping(t):
    if not t:
        return t
    token_mapping, _ = _build_combined_mapping()
    return normalize_space(" ".join(token_mapping.get(x, x) for x in t.split()))


def apply_all_dictionaries(t):
    t = apply_phrase_mapping(t)
    t = apply_token_mapping(t)
    t = normalize_repeated_characters_fallback(t)
    t = apply_phrase_mapping(t)
    t = apply_token_mapping(t)
    return normalize_space(t)


def basic_clean_review(text):
    """Pipeline lengkap: basic_text_clean -> normalisasi kamus (2 pass)."""
    t = basic_text_clean(text)
    if not t:
        return ""
    return normalize_space(apply_all_dictionaries(t))


def preprocess_with_trace(text):
    """Sama seperti basic_clean_review, tapi mengembalikan tiap tahap (buat ditampilkan di UI)."""
    step0 = text
    step1 = basic_text_clean(text)
    step2 = apply_all_dictionaries(step1) if step1 else ""
    word_count = len(step2.split())
    return {
        "asli": step0,
        "setelah_basic_clean": step1,
        "setelah_normalisasi_kamus": step2,
        "jumlah_kata": word_count,
    }
