"""
LDA topic assignment untuk teks baru — replikasi persis preprocessing token notebook
06a_topic_modeling_subaspek.ipynb (tokenisasi -> stopword removal -> stemming Sastrawi ->
bigram/trigram -> buang token brand), lalu dominant-topic lookup pakai dictionary + model
LDA yang sudah di-generate ulang per aspek (lihat assets/lda/<Aspek>/).
"""
import re
from pathlib import Path
from functools import lru_cache

from topic_names import nama_subtopik

# gensim & Sastrawi diimpor MALAS di dalam fungsi (lihat catatan optimasi di docstring)

ASSETS_DIR = Path(__file__).parent / "assets" / "lda"
ASPECTS = ["Individual", "Technical", "Social", "Financial"]

# --- stopwords (persis notebook 06a) ---------------------------------------
_STOPWORD_TAMBAHAN = {
    "bagus", "bantu", "suka", "mantap", "terimakasih", "alhamdulillah", "aplikasi",
    "apk", "app", "nya", "yg", "ga", "gak", "sih", "deh", "dong", "kok", "ya", "aja",
    "banget", "bgt", "semoga", "terima", "kasih", "tolong", "mohon",
    "aku", "jadi", "kalau", "buat", "mau", "baik", "lebih", "terus", "sekali",
    "banyak", "sama", "tahu", "dulu", "padahal", "coba", "apa", "malah",
    "selalu", "sekarang", "pertama", "awal", "dapat", "tambah", "nambah", "pokok",
    "sangat", "amat", "terlalu",
}
_KATA_NEGASI = {"tidak", "bukan", "belum", "jangan", "tak", "kurang"}


@lru_cache(maxsize=1)
def _get_stopwords():
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    stopword_dasar = set(StopWordRemoverFactory().get_stop_words())
    return (stopword_dasar | _STOPWORD_TAMBAHAN) - _KATA_NEGASI

_STEM_EXCEPTION = {"berbagi": "berbagi"}
TOKEN_BRAND = {"theasianparent", "asianparent", "hallobumil", "temanbumil", "bukubumil"}


def tokenisasi_awal(teks):
    stopwords = _get_stopwords()
    teks = re.sub(r"[^a-zA-Z\s]", " ", str(teks).lower())
    tokens = teks.split()
    return [t for t in tokens if t not in stopwords and len(t) > 2]


@lru_cache(maxsize=1)
def _get_stemmer():
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    return StemmerFactory().create_stemmer()


@lru_cache(maxsize=20000)
def _stem_kata(kata):
    """Hasil stemming per kata di-cache — kata yang sama tidak distem ulang.
    Hasil untuk sebuah kata selalu sama, jadi caching tidak mengubah keluaran."""
    if kata in _STEM_EXCEPTION:
        return _STEM_EXCEPTION[kata]
    return _get_stemmer().stem(kata)


def stem_tokens(tokens):
    return [_stem_kata(t) for t in tokens]


@lru_cache(maxsize=1)
def _load_phrasers():
    from gensim.models.phrases import Phraser
    bigram = Phraser.load(str(ASSETS_DIR / "bigram_phraser.gensim"))
    trigram = Phraser.load(str(ASSETS_DIR / "trigram_phraser.gensim"))
    return bigram, trigram


@lru_cache(maxsize=4)
def _load_lda(aspek):
    from gensim import corpora
    from gensim.models import LdaModel
    aspek_dir = ASSETS_DIR / aspek
    dictionary = corpora.Dictionary.load(str(aspek_dir / "dictionary.gensim"))
    model = LdaModel.load(str(aspek_dir / "lda_model.gensim"))
    return dictionary, model


def teks_ke_token_ngram(teks_sudah_bersih):
    """Input: teks HASIL preprocessing (basic_clean_review), bukan teks mentah."""
    tokens = tokenisasi_awal(teks_sudah_bersih)
    tokens = stem_tokens(tokens)
    bigram, trigram = _load_phrasers()
    tokens_ngram = trigram[bigram[tokens]]
    tokens_ngram = [t for t in tokens_ngram if t not in TOKEN_BRAND]
    return tokens_ngram


def prediksi_topik(teks_sudah_bersih, aspek, topn_kata=8):
    """
    Kembalikan dominant topic untuk 1 aspek yang terdeteksi pada 1 review.
    Return None kalau token terlalu sedikit / tidak ada kata yang dikenali dictionary aspek itu.
    """
    if aspek not in ASPECTS:
        raise ValueError(f"Aspek tidak dikenal: {aspek}")

    tokens_ngram = teks_ke_token_ngram(teks_sudah_bersih)
    if len(tokens_ngram) == 0:
        return None

    dictionary, model = _load_lda(aspek)
    bow = dictionary.doc2bow(tokens_ngram)
    if len(bow) == 0:
        return None

    topik_probs = model.get_document_topics(bow, minimum_probability=0)
    dominant_topic_id, dominant_prob = max(topik_probs, key=lambda x: x[1])

    kata_representatif = [w.replace("_", " ") for w, _ in model.show_topic(dominant_topic_id, topn=topn_kata)]

    return {
        "aspek": aspek,
        "topic_id": dominant_topic_id,
        "probabilitas": float(dominant_prob),
        "nama_subtopik": nama_subtopik(aspek, dominant_topic_id),
        "kata_representatif": kata_representatif,
        "semua_topik_prob": sorted(
            [(tid, float(p), nama_subtopik(aspek, tid)) for tid, p in topik_probs],
            key=lambda x: -x[1],
        ),
    }
