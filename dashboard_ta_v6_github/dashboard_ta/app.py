import pandas as pd
import plotly.express as px
import streamlit as st

import charts as ch
import data as d
import pipeline as pipe
import ui

st.set_page_config(
    page_title="Dashboard TA — ABSA Aplikasi Kesehatan Ibu Hamil",
    page_icon="🤰",
    layout="wide",
    initial_sidebar_state="expanded",
)
ui.inject_css()

ASPECT_ORDER = ["Individual", "Technical", "Social", "Financial"]
SENTIMENT_ORDER = ["Positif", "Netral", "Negatif"]
CY = ui.CYAN
style_fig = ch.style_fig
rb = ch.rb

SENT_LABEL_COLOR = {
    "positif": (d.COLOR["good"], "#eefaee", "#bfe6bf"),
    "negatif": (d.COLOR["critical"], "#fdeeee", "#f3c9c9"),
    "netral": (d.COLOR["warning"], "#fff8e8", "#f5e0b0"),
}


@st.cache_data(show_spinner=False, max_entries=64)
def analisis_cached(teks_tuple, mode_lengkap, _progress_cb=None):
    """Hasil analisis di-cache: menganalisis komentar yang sama dua kali jadi instan.
    Argumen berawalan '_' tidak ikut jadi kunci cache (aturan Streamlit)."""
    return pipe.analisis_banyak(list(teks_tuple), mode_lengkap, progress_cb=_progress_cb)


# ===========================================================================
# SIDEBAR — 3 menu utama
# ===========================================================================
with st.sidebar:
    st.markdown(
        """<div class="side-brand">
            <div class="side-brand-title">🤰 Dashboard Tugas Akhir</div>
            <div class="side-brand-sub">Aspect-Based Sentiment Analysis<br>Ulasan Aplikasi Kesehatan Ibu Hamil</div>
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="side-label">Menu</div>', unsafe_allow_html=True)
    MENU = ["🏠  Dashboard Utama", "📚  Detail Penelitian", "🔮  Prediksi Komentar"]
    menu = st.radio("Menu", MENU, label_visibility="collapsed")

    SUB_PAGES = [
        "📥  Data & Preprocessing",
        "🏷️  Gold Standard & Reliabilitas",
        "🤖  Pelabelan Otomatis (XNLI)",
        "📈  Performa Model DistilBERT",
        "🧩  Topic Modeling (LDA)",
        "✅  Validasi Ahli",
        "📌  Kesimpulan & Insight",
    ]
    sub_page = SUB_PAGES[0]
    if menu == MENU[1]:
        st.markdown('<div class="side-label">Sub-menu</div>', unsafe_allow_html=True)
        sub_page = st.radio("Sub-menu", SUB_PAGES, label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"""<div class="side-foot">
            <b>{d.META['penulis']}</b><br>{d.META['nim']}<br><br>
            {d.META['prodi']}<br>{d.META['fakultas']}<br>{d.META['universitas']}, {d.META['tahun']}
            <hr style="border:none;border-top:1px solid {d.COLOR['grid']};margin:9px 0;">
            <b>Pembimbing</b><br>1. {d.META['pembimbing1']}<br>2. {d.META['pembimbing2']}
        </div>""",
        unsafe_allow_html=True,
    )


# ===========================================================================
# MENU 1 — DASHBOARD UTAMA
# ===========================================================================
if menu == MENU[0]:
    F = ch.dashboard()

    ui.hero(
        "Aspect-Based Sentiment Analysis",
        "Menggali aspek dan sentimen dari ulasan pengguna aplikasi kesehatan ibu hamil (mHealth) "
        "menggunakan Zero-shot Classification (XNLI) dan DistilBERT, dilengkapi eksplorasi "
        "sub-topik dengan Topic Modeling (LDA).",
        kicker="Dashboard Utama · Tugas Akhir 2026",
        chips=[
            f"👤 {d.META['penulis']}",
            f"🎓 {d.META['prodi']}",
            f"🏛️ {d.META['universitas']}",
            "🧭 Kerangka KDD",
        ],
    )

    ui.section("Ringkasan Angka Utama", "Indikator kunci dari keseluruhan tahapan penelitian")
    c1, c2, c3, c4 = st.columns(4)
    ui.kpi(c1, "Ulasan Dianalisis", rb(d.KPI["total_ulasan_final"]),
           f"dari {rb(d.KPI['total_ulasan_scraped'])} hasil scraping", CY["600"], "💬")
    ui.kpi(c2, "Aplikasi mHealth", d.KPI["jumlah_aplikasi"], "sumber: Google Play Store",
           d.COLOR["orange"], "📱")
    ui.kpi(c3, "Kategori Aspek", d.KPI["jumlah_aspek"], "Individual · Technical · Social · Financial",
           d.COLOR["aqua"], "🧩")
    ui.kpi(c4, "Sub-Topik LDA", d.KPI["jumlah_subtopik"], "hasil eksplorasi topic modeling",
           d.COLOR["yellow"], "🔎")

    st.write("")
    c5, c6, c7, c8 = st.columns(4)
    ui.kpi(c5, "F1 Weighted Aspek", f"{d.KPI['f1_weighted_aspek']:.4f}", "DistilBERT · rasio 80:10:10",
           CY["700"], "🎯")
    ui.kpi(c6, "F1 Weighted Sentimen", f"{d.KPI['f1_weighted_sentimen']:.4f}", "DistilBERT · rasio 60:20:20",
           d.COLOR["good"], "🎯")
    ui.kpi(c7, "Kappa Aspek", f"{d.KPI['kappa_aspek_avg']:.3f}", "kesepakatan anotator · Substantial",
           d.COLOR["violet"], "🤝")
    ui.kpi(c8, "Kappa Sentimen", f"{d.KPI['kappa_sentimen']:.3f}", "kesepakatan anotator · Substantial",
           d.COLOR["magenta"], "🤝")

    st.write("")
    st.divider()

    ui.section("Sekilas Hasil", "Pilih tampilan untuk melihat gambaran besar hasil penelitian")
    pilih = st.radio(
        "Tampilan", ["Distribusi Aspek", "Distribusi Sentimen", "Performa Model"],
        horizontal=True, label_visibility="collapsed",
    )

    if pilih == "Distribusi Aspek":
        c_a, c_b = st.columns([1.3, 1])
        c_a.plotly_chart(F["aspek_dist"], use_container_width=True)
        c_b.plotly_chart(F["aspek_aktif"], use_container_width=True)
        ui.info(
            "Aspek <b>Individual</b> paling dominan (84,13% dari ulasan terlabel) — pengguna paling "
            "banyak membicarakan manfaat dan pengalaman pribadi memakai aplikasi. Sementara "
            "<b>10.072</b> ulasan tidak memiliki aspek aktif sama sekali (NO_ASPECT)."
        )
    elif pilih == "Distribusi Sentimen":
        c_a, c_b = st.columns([1, 1.4])
        c_a.plotly_chart(F["sent_pie"], use_container_width=True)
        c_b.plotly_chart(F["sent_stack"], use_container_width=True)
        ui.info(
            "Sentimen <b>positif</b> mendominasi keseluruhan ulasan, namun komposisinya berbeda "
            "antar aspek — aspek Technical dan Financial memuat proporsi keluhan yang jauh lebih tinggi."
        )
    else:
        tab_a, tab_s = st.tabs(["Model Aspek", "Model Sentimen"])
        with tab_a:
            c_a, c_b = st.columns([1.4, 1])
            c_a.plotly_chart(F["ratio_aspek"], use_container_width=True)
            c_b.plotly_chart(F["f1_aspek"], use_container_width=True)
            st.success(f"Rasio terbaik: **{d.BEST_RATIO_ASPECT}** — F1 Weighted (Gold) = 0,7098")
        with tab_s:
            c_a, c_b = st.columns([1.4, 1])
            c_a.plotly_chart(F["ratio_sent"], use_container_width=True)
            c_b.plotly_chart(F["f1_sent"], use_container_width=True)
            st.success(f"Rasio terbaik: **{d.BEST_RATIO_SENTIMENT}** — F1 Weighted (Gold) = 0,8631")

    st.divider()
    ui.section("Tahapan Metodologi", "Kerangka Knowledge Discovery in Databases (KDD)", d.COLOR["aqua"])
    warna_kdd = [CY["700"], CY["600"], CY["500"], d.COLOR["aqua"], d.COLOR["yellow"], d.COLOR["orange"]]
    kdd_cols = st.columns(3)
    for i, (title, desc) in enumerate(d.KDD_STAGES):
        with kdd_cols[i % 3]:
            ui.card(title, desc, accent=warna_kdd[i % len(warna_kdd)], icon="▸", equal_height=True)

    st.divider()
    ui.section("Jelajahi Dashboard", "Gunakan menu di sidebar untuk membuka bagian berikut")
    n1, n2, n3 = st.columns(3)
    ui.navcard(n1, "📚", "Detail Penelitian",
               "Tujuh sub-halaman berisi data & preprocessing, gold standard, pelabelan otomatis, "
               "performa model, topic modeling, validasi ahli, dan kesimpulan.", CY["600"])
    ui.navcard(n2, "🔮", "Prediksi Komentar",
               "Uji langsung model asli hasil penelitian ini pada komentar yang kamu tulis sendiri — "
               "satu komentar atau sekaligus banyak (batch).", d.COLOR["orange"])
    ui.navcard(n3, "🧠", "Model yang Dipakai",
               "Zero-shot Classification (mDeBERTa XNLI), DistilBERT Indonesia hasil fine-tuning, "
               "dan LDA (gensim) untuk sub-topik per aspek.", d.COLOR["aqua"])

    st.write("")
    with st.expander("📄 Abstrak Penelitian"):
        st.write(d.META["abstrak"])


# ===========================================================================
# MENU 2 — DETAIL PENELITIAN
# ===========================================================================
elif menu == MENU[1]:
    T = ch.tables()

    # ---------------------------------------------------------------- 2.1
    if sub_page == SUB_PAGES[0]:
        F = ch.data_prep()
        ui.hero("Data & Preprocessing",
                "Sumber data, alur penyaringan, dan tahapan normalisasi teks sebelum masuk ke model.",
                kicker="Detail Penelitian · 1 dari 7")

        ui.section("Data Selection",
                   "Ulasan dikumpulkan dari 10 aplikasi kesehatan ibu hamil di Google Play Store")
        st.plotly_chart(F["apps"], use_container_width=True)
        with st.expander("📋 Lihat tabel rinci per aplikasi"):
            st.dataframe(T["apps"], hide_index=True, use_container_width=True, height=386)

        st.divider()
        ui.section("Alur Penyaringan Data", "Penyusutan jumlah ulasan dari scraping hingga siap dianalisis",
                   d.COLOR["orange"])
        c1, c2 = st.columns(2)
        c1.plotly_chart(F["funnel_prep"], use_container_width=True)
        c2.plotly_chart(F["funnel_filter"], use_container_width=True)

        ui.info(
            f"Dari <b>100.109</b> ulasan hasil scraping, tersisa <b>68.659</b> setelah data preparation, "
            f"dan akhirnya <b>{rb(d.TOTAL_ULASAN_PREPROCESSED)}</b> ulasan siap dianalisis setelah "
            f"penghapusan ulasan kurang dari 3 kata.",
            "success",
        )

        st.divider()
        ui.section("Tahapan Text Preprocessing", "Sepuluh tahap normalisasi yang diterapkan berurutan",
                   d.COLOR["aqua"])
        warna_step = [CY["700"], CY["600"], CY["500"], d.COLOR["aqua"], d.COLOR["yellow"]]
        cols = st.columns(5)
        for i, step in enumerate(d.PREP_STEPS):
            with cols[i % 5]:
                st.markdown(
                    f'<div class="card card-accent" style="--accent:{warna_step[i % 5]}; '
                    f'text-align:center; min-height:104px;">'
                    f'<div style="color:{warna_step[i % 5]}; font-weight:800; font-size:1.25rem;">{i+1}</div>'
                    f'<div class="card-desc" style="font-weight:600; color:{d.COLOR["ink"]};">{step}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ---------------------------------------------------------------- 2.2
    elif sub_page == SUB_PAGES[1]:
        F = ch.gold()
        ui.hero("Gold Standard & Reliabilitas",
                "Pembentukan data acuan manual oleh dua anotator, pengukuran kesepakatan, "
                "dan optimasi threshold zero-shot.",
                kicker="Detail Penelitian · 2 dari 7")

        ui.section("Pembentukan Gold Standard",
                   "500 ulasan disampling (acak + terarah berbasis kata kunci) lalu dianotasi manual")
        c1, c2, c3 = st.columns([1, 1, 1.15])
        c1.plotly_chart(F["sampling"], use_container_width=True)
        c2.plotly_chart(F["support"], use_container_width=True)
        c3.plotly_chart(F["gold_sent"], use_container_width=True)

        st.divider()
        ui.section("Reliabilitas Antar-Anotator (Cohen's Kappa)",
                   "Mengukur tingkat kesepakatan dua anotator saat membentuk gold standard",
                   d.COLOR["violet"])
        c1, c2 = st.columns([1.4, 1])
        c1.plotly_chart(F["kappa"], use_container_width=True)
        with c2:
            st.markdown("**Skala Interpretasi Kappa**")
            st.dataframe(d.KAPPA_SCALE, hide_index=True, use_container_width=True, height=245)
            st.metric("Kappa Sentimen (Gold)", f"{d.KAPPA_SENTIMENT['nilai']:.3f}",
                      d.KAPPA_SENTIMENT["interpretasi"])

        st.divider()
        ui.section("Optimasi Threshold Zero-shot (Aspek)",
                   "Perbandingan skema hipotesis bawaan (A) vs custom (B), dan threshold optimal hasil grid search",
                   d.COLOR["orange"])
        c1, c2 = st.columns(2)
        c1.plotly_chart(F["auc"], use_container_width=True)
        c2.plotly_chart(F["threshold"], use_container_width=True)

    # ---------------------------------------------------------------- 2.3
    elif sub_page == SUB_PAGES[2]:
        F = ch.xnli()
        ui.hero("Pelabelan Otomatis (XNLI)",
                "Hasil automatic labeling aspek & sentimen dengan Zero-shot Classification, "
                "beserta kesesuaiannya terhadap gold standard.",
                kicker="Detail Penelitian · 3 dari 7")

        ui.section("Distribusi Hasil Pelabelan Aspek",
                   f"Zero-shot Classification pada {rb(d.TOTAL_ULASAN_PREPROCESSED)} ulasan")
        c1, c2 = st.columns(2)
        c1.plotly_chart(F["aktif"], use_container_width=True)
        c2.plotly_chart(F["aspek_dist"], use_container_width=True)
        ui.info(
            "Dari 59.620 ulasan, <b>10.072</b> tidak memiliki aspek aktif (NO_ASPECT). Aspek "
            "<b>Individual</b> paling dominan (84,13% dari ulasan terlabel), diikuti Technical (28,10%), "
            "Social (8,40%), dan Financial (6,01%)."
        )

        st.divider()
        ui.section("Distribusi Hasil Pelabelan Sentimen",
                   f"Pelabelan pada {rb(d.TOTAL_PASANGAN_ASPEK_SENTIMEN)} pasangan aspek-ulasan",
                   d.COLOR["aqua"])
        c1, c2 = st.columns([1, 1.4])
        c1.plotly_chart(F["sent_pie"], use_container_width=True)
        c2.plotly_chart(F["sent_stack"], use_container_width=True)

        with st.expander("📋 Tabel silang aspek × sentimen"):
            st.dataframe(T["cross"], hide_index=True, use_container_width=True)

        st.divider()
        ui.section("Kesesuaian XNLI terhadap Gold Standard", accent=d.COLOR["violet"])
        tab1, tab2 = st.tabs(["Aspek", "Sentimen"])
        with tab1:
            c1, c2 = st.columns([1.3, 1])
            c1.plotly_chart(F["gold_match"], use_container_width=True)
            with c2:
                st.dataframe(d.df_gold_match_aspect, hide_index=True, use_container_width=True, height=180)
                m1, m2, m3 = st.columns(3)
                m1.metric("F1 Macro", f"{d.ASPECT_LABELING_SUMMARY['f1_macro']:.4f}")
                m2.metric("F1 Weighted", f"{d.ASPECT_LABELING_SUMMARY['f1_weighted']:.4f}")
                m3.metric("Kappa#2 Avg", f"{d.ASPECT_LABELING_SUMMARY['kappa2_avg']:.4f}")
        with tab2:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Sentimen Murni (n=378)**")
                st.dataframe(T["sent_murni"], hide_index=True, use_container_width=True)
            with c2:
                st.markdown("**End-to-end (n=202)**")
                st.dataframe(T["sent_e2e"], hide_index=True, use_container_width=True)
            st.write("")
            st.plotly_chart(F["sent_per_aspek"], use_container_width=True)

    # ---------------------------------------------------------------- 2.4
    elif sub_page == SUB_PAGES[3]:
        F = ch.performa()
        ui.hero("Performa Model DistilBERT",
                "Perbandingan tiga rasio pembagian data, classification report terhadap gold standard, "
                "dan eksperimen threshold tuning.",
                kicker="Detail Penelitian · 4 dari 7")

        ui.section("Pengaruh Rasio Pembagian Data", "Tiga skenario rasio train : validation : test")
        tab1, tab2 = st.tabs(["Model Klasifikasi Aspek", "Model Klasifikasi Sentimen"])
        with tab1:
            st.plotly_chart(F["ratio_aspek"], use_container_width=True)
            st.success(f"**Rasio terbaik: {d.BEST_RATIO_ASPECT}** — F1 Weighted (Gold) = 0,7098")
            st.dataframe(d.df_ratio_aspect, hide_index=True, use_container_width=True)
        with tab2:
            st.plotly_chart(F["ratio_sent"], use_container_width=True)
            st.success(f"**Rasio terbaik: {d.BEST_RATIO_SENTIMENT}** — F1 Weighted (Gold) = 0,8631")
            st.dataframe(d.df_ratio_sentiment, hide_index=True, use_container_width=True)

        st.divider()
        ui.section("Classification Report vs Gold Standard",
                   "Model dengan rasio pembagian data terbaik masing-masing", d.COLOR["aqua"])
        tab1, tab2 = st.tabs(["Aspek (rasio 80:10:10)", "Sentimen (rasio 60:20:20)"])
        with tab1:
            c1, c2 = st.columns([1.3, 1])
            c1.plotly_chart(F["aspek_prf"], use_container_width=True)
            c2.dataframe(T["aspek_gold"], hide_index=True, use_container_width=True, height=280)
            ui.info(
                "Aspek <b>Individual</b> (F1 0,874) dan <b>Social</b> (F1 0,864) menunjukkan performa "
                "tertinggi, sedangkan <b>Technical</b> (F1 0,476) dan <b>Financial</b> (F1 0,407) terendah — "
                "konsisten dengan kualitas pseudo-label XNLI yang lebih rendah pada kedua aspek tersebut.",
                "warning",
            )
            st.markdown("**Eksperimen Threshold Tuning (Post-hoc)**")
            st.plotly_chart(F["threshold_delta"], use_container_width=True)
            m1, m2 = st.columns([1, 2])
            m1.metric("F1 Macro (Baseline → Tuned)", f"{d.THRESHOLD_TUNING_MACRO['tuned']:.4f}",
                      f"+{d.THRESHOLD_TUNING_MACRO['delta']:.4f}")
            m2.dataframe(d.df_threshold_tuning, hide_index=True, use_container_width=True)
        with tab2:
            c1, c2 = st.columns([1.3, 1])
            c1.plotly_chart(F["sent_prf"], use_container_width=True)
            c2.dataframe(T["sent_gold"], hide_index=True, use_container_width=True, height=280)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Analisis Kesalahan Prediksi Kelas Netral**")
                st.plotly_chart(F["netral_err"], use_container_width=True)
            with c2:
                st.markdown("**Classification Report — Test Set Besar (n=12.431)**")
                st.dataframe(T["sent_test"], hide_index=True, use_container_width=True)
                ui.info("Model cenderung memprediksi data netral sebagai <b>positif (60%)</b>, "
                        "mengindikasikan kelas netral sulit dibedakan akibat jumlah data yang sangat minoritas.",
                        "warning")

        st.divider()
        with st.expander("⚙️ Hyperparameter Fine-tuning DistilBERT"):
            st.dataframe(d.TRAIN_PARAMS, hide_index=True, use_container_width=True)

    # ---------------------------------------------------------------- 2.5
    elif sub_page == SUB_PAGES[4]:
        F = ch.lda()
        ui.hero("Topic Modeling (LDA)",
                "Eksplorasi sub-topik di dalam tiap kategori aspek menggunakan Latent Dirichlet Allocation.",
                kicker="Detail Penelitian · 5 dari 7")

        ui.section("Konstruksi Model LDA", "Ukuran korpus dan coherence score pada K final tiap aspek")
        c1, c2 = st.columns(2)
        c1.plotly_chart(F["corpus"], use_container_width=True)
        c2.plotly_chart(F["coherence"], use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        for col, (_, row) in zip([c1, c2, c3, c4], d.df_lda_coherence.iterrows()):
            ui.kpi(col, row["Aspek"], f"K = {row['K Final']}",
                   f"coherence {row['Coherence (K Final)']:.4f}",
                   d.ASPECT_COLOR.get(row["Aspek"], CY["500"]), "🧩")

        st.divider()
        ui.section(f"Distribusi {d.TOTAL_SUBTOPICS} Sub-Topik",
                   "Ukuran blok merepresentasikan jumlah dokumen pada tiap sub-topik "
                   "(klik blok untuk memperbesar)", d.COLOR["orange"])
        st.plotly_chart(F["treemap"], use_container_width=True)

        aspek_filter = st.selectbox("Filter tabel sub-topik berdasarkan aspek", ["Semua"] + ASPECT_ORDER)
        st.dataframe(ch.subtopik_table(aspek_filter), hide_index=True, use_container_width=True)

    # ---------------------------------------------------------------- 2.6
    elif sub_page == SUB_PAGES[5]:
        F = ch.validasi()
        ui.hero("Validasi Ahli",
                "Penilaian dua validator industri terhadap relevansi problem owner, kategori aspek, "
                "dan kejelasan label sub-topik.",
                kicker="Detail Penelitian · 6 dari 7")

        ui.section("Profil Validator")
        vc1, vc2 = st.columns(2)
        warna_v = [CY["600"], d.COLOR["orange"]]
        for i, (col, (_, row)) in enumerate(zip([vc1, vc2], d.df_validators.iterrows())):
            with col:
                st.markdown(
                    f'<div class="card card-accent" style="--accent:{warna_v[i]};">'
                    f'<div class="card-title">👤 {row["Validator"]} — {row["Nama"]}</div>'
                    f'<div class="card-desc">{row["Profesi/Jabatan"]}<br>{row["Institusi"]}<br>'
                    f'<span style="color:{d.COLOR["muted"]};">Pengalaman {row["Pengalaman (tahun)"]} tahun · '
                    f'Validasi {row["Tanggal Validasi"]}</span></div></div>',
                    unsafe_allow_html=True,
                )

        st.divider()
        ui.section("Penilaian Relevansi Problem Owner & Kategori Aspek",
                   "Skala 1 (sangat tidak setuju) – 5 (sangat setuju)", d.COLOR["aqua"])
        c1, c2 = st.columns([1.5, 1])
        c1.plotly_chart(F["stmt"], use_container_width=True)
        c2.dataframe(d.df_validation_statements[["No", "Validator 1", "Validator 2", "Rata-rata"]],
                     hide_index=True, use_container_width=True, height=225)
        with st.expander("📄 Lihat teks lengkap pernyataan"):
            for _, r in d.df_validation_statements.iterrows():
                st.markdown(f"**{r['No']}.** {r['Pernyataan']}")

        st.divider()
        ui.section("Relevansi & Kejelasan Label Sub-Topik per Aspek", accent=d.COLOR["violet"])
        st.plotly_chart(F["subtopic"], use_container_width=True)
        ui.info(
            "Aspek <b>Social</b> memperoleh skor tertinggi (4,00 relevansi & kejelasan), sementara "
            "<b>Financial</b> memperoleh skor relevansi terendah (2,42), sejalan dengan performa "
            "klasifikasi yang juga rendah pada aspek ini."
        )
        st.dataframe(d.df_validation_subtopic, hide_index=True, use_container_width=True)

        st.divider()
        ui.section("Penilaian Kesesuaian Kategori Aspek", "Ya / Sebagian / Tidak", d.COLOR["yellow"])
        st.dataframe(d.df_validation_yesno, hide_index=True, use_container_width=True)

    # ---------------------------------------------------------------- 2.7
    else:
        F = ch.kesimpulan()
        ui.hero("Kesimpulan & Insight",
                "Rangkuman temuan utama, gap performa antar sumber evaluasi, dan keterkaitan antar hasil.",
                kicker="Detail Penelitian · 7 dari 7")

        ui.section("Ringkasan Temuan Utama (Knowledge)", "Rangkuman hasil berdasarkan tahapan KDD")
        warna_k = [CY["700"], CY["600"], CY["500"], d.COLOR["aqua"], d.COLOR["yellow"], d.COLOR["orange"],
                   d.COLOR["violet"], d.COLOR["magenta"]]
        for i, (comp, stage, finding) in enumerate(d.KNOWLEDGE_SUMMARY):
            st.markdown(
                f'<div class="card card-accent" style="--accent:{warna_k[i % len(warna_k)]};">'
                f'<div class="card-title">{comp}'
                f'<span style="color:{d.COLOR["muted"]}; font-weight:500; font-size:0.78rem;">— {stage}</span>'
                f'</div><div class="card-desc">{finding}</div></div>',
                unsafe_allow_html=True,
            )

        st.divider()
        ui.section("Perbandingan Rasio Terbaik: Model Aspek vs Model Sentimen", accent=d.COLOR["aqua"])
        st.dataframe(T["ratio_summary"], hide_index=True, use_container_width=True)
        ui.info(
            "Rasio pembagian data optimal <b>berbeda antar model</b>: 80:10:10 untuk klasifikasi aspek, "
            "60:20:20 untuk klasifikasi sentimen — namun selisih performa antar-rasio relatif kecil "
            "(&lt; 0,015), menunjukkan model DistilBERT cukup stabil terhadap variasi rasio split.",
            "success",
        )

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            ui.section("Gap Performa: Data Test vs Gold Standard", accent=d.COLOR["orange"])
            st.plotly_chart(F["gap"], use_container_width=True)
            st.dataframe(T["gap"], hide_index=True, use_container_width=True)
        with c2:
            ui.section("Keterkaitan F1, Sub-Topik & Coherence", accent=d.COLOR["violet"])
            st.plotly_chart(F["korelasi"], use_container_width=True)
            st.dataframe(T["korelasi"], hide_index=True, use_container_width=True)

        ui.info(
            "Aspek dengan performa klasifikasi lebih tinggi (Individual, Social) cenderung memiliki "
            "jumlah sub-topik LDA lebih sedikit namun lebih koheren, sementara Technical dan Financial — "
            "dengan performa klasifikasi rendah — menunjukkan sub-topik lebih banyak dan tumpang tindih, "
            "mengindikasikan kompleksitas isu yang lebih tinggi."
        )
        st.caption(
            f"Dashboard ini dibangun berdasarkan hasil Tugas Akhir \"{d.META['judul']}\" — "
            f"{d.META['penulis']} ({d.META['nim']}), {d.META['prodi']}, {d.META['fakultas']}, "
            f"{d.META['universitas']} {d.META['tahun']}."
        )


# ===========================================================================
# MENU 3 — PREDIKSI KOMENTAR
# ===========================================================================
else:
    ui.hero(
        "Prediksi Komentar",
        "Jalankan langsung model asli hasil penelitian ini terhadap komentar yang kamu tulis sendiri — "
        "bukan simulasi maupun angka statis.",
        kicker="Live Inference",
        chips=["🤖 DistilBERT Aspek", "💬 DistilBERT Sentimen", "🧩 LDA Sub-Topik", "⚖️ Zero-shot XNLI"],
    )

    # Mulai memuat model di latar belakang sambil pengguna membaca / mengetik komentar,
    # supaya saat tombol ditekan model sudah siap dipakai.
    if not st.session_state.get("_prewarm"):
        st.session_state["_prewarm"] = True
        pipe.prewarm_async()

    with st.expander("ℹ️  Bagaimana cara kerjanya?", expanded=False):
        s1, s2 = st.columns(2)
        with s1:
            ui.card("1. Preprocessing",
                    "Teks dibersihkan (normalisasi unicode, hapus URL/email/simbol) lalu dinormalisasi "
                    "memakai kamus typo &amp; singkatan hasil penelitian.", CY["700"], "🧹")
            ui.card("2. Deteksi Aspek",
                    "Model <b>DistilBERT</b> multi-label menentukan aspek mana yang dibahas: Individual, "
                    "Technical, Social, dan/atau Financial (bisa lebih dari satu, bisa juga tidak ada).",
                    CY["600"], "🤖")
            ui.card("3. Sentimen per Aspek",
                    "Untuk tiap aspek yang terdeteksi, model <b>DistilBERT</b> sentimen menentukan "
                    "penilaiannya: positif, negatif, atau netral.", CY["500"], "💬")
        with s2:
            ui.card("4. Topic Modeling (mode Lengkap)",
                    "Model <b>LDA</b> mencari sub-topik paling cocok di dalam aspek tersebut, misalnya "
                    "\"Error dan Aplikasi Lambat Setelah Update\" untuk aspek Technical.",
                    d.COLOR["aqua"], "🧩")
            ui.card("5. Pembanding Zero-shot (mode Lengkap)",
                    "Skor <b>XNLI</b> — model zero-shot yang dipakai melabeli 59.620 data latih secara "
                    "otomatis — ditampilkan berdampingan agar perbedaannya terlihat.",
                    d.COLOR["orange"], "⚖️")
            ui.info(
                "Model DistilBERT (±260&nbsp;MB × 2) diunduh otomatis sekali saat pertama dipakai. "
                "Tokenizer dasar &amp; model XNLI diunduh dari HuggingFace Hub — pastikan komputer "
                "terhubung internet.", "warning",
            )

    # ---------------- form input ----------------
    ui.section("Masukkan Komentar", "Pilih mode analisis, lalu tulis komentar yang ingin diuji")

    cfg1, cfg2 = st.columns([1.1, 1])
    with cfg1:
        mode = st.radio(
            "Kedalaman analisis",
            ["⚡ Cepat — DistilBERT saja", "🔬 Lengkap — + Topic Modeling & Pembanding XNLI"],
        )
        mode_lengkap = mode.startswith("🔬")
    with cfg2:
        jumlah = st.radio("Jumlah komentar", ["📝 Satu komentar", f"📚 Banyak komentar (maks {pipe.MAX_BATCH})"])
        batch_mode = jumlah.startswith("📚")

    ui.info(
        f"Batas input: maksimal <b>{pipe.MAX_CHAR} karakter</b> per komentar"
        + (f", dan maksimal <b>{pipe.MAX_BATCH} komentar</b> sekali jalan." if batch_mode else ".")
        + f" Komentar dengan kurang dari <b>{pipe.MIN_KATA} kata</b> (setelah preprocessing) akan dilewati, "
        "sesuai aturan filtering di penelitian ini."
        + ("<br><b>Mode Lengkap</b> memakai model XNLI berukuran besar, sehingga tiap komentar "
           "butuh waktu lebih lama." if mode_lengkap else "")
    )

    CONTOH = {
        "— pilih contoh —": "",
        "😊 Positif · Individual": "Aplikasinya sangat membantu aku memantau kehamilan, fitur lengkap dan mudah dipakai",
        "😠 Negatif · Technical": "Kenapa aplikasi ini sering force close ya, udah update masih aja error pas dibuka",
        "🤝 Positif · Social": "Aku suka banget bisa sharing pengalaman sama ibu hamil lain di forum komunitasnya",
        "💸 Negatif · Financial": "Fitur premiumnya mahal banget, harusnya yang penting-penting gratis aja",
    }

    if batch_mode:
        cc1, cc2 = st.columns([1, 1])
        with cc1:
            contoh_pilih = st.selectbox("Sisipkan contoh (opsional)", list(CONTOH.keys()))
        with cc2:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            st.caption("💡 Tulis **satu komentar per baris**. Baris kosong otomatis diabaikan.")

        default_batch = "\n".join([v for k, v in CONTOH.items() if v]) \
            if contoh_pilih == "— pilih contoh —" else CONTOH[contoh_pilih]
        teks_batch = st.text_area(
            f"Daftar komentar (satu per baris, maksimal {pipe.MAX_BATCH} baris)",
            value=default_batch, height=190,
            placeholder="aplikasinya bagus tapi sering error\nfitur premium kemahalan\n...",
        )
        daftar = pipe.parse_batch(teks_batch)
        terlalu_panjang = [i + 1 for i, t in enumerate(daftar) if len(t) > pipe.MAX_CHAR]

        m1, m2, m3 = st.columns(3)
        ui.kpi(m1, "Komentar Terdeteksi", f"{len(daftar)}", f"batas {pipe.MAX_BATCH} komentar",
               CY["600"] if len(daftar) <= pipe.MAX_BATCH else d.COLOR["critical"], "📚")
        ui.kpi(m2, "Karakter Terpanjang", f"{max((len(t) for t in daftar), default=0)}",
               f"batas {pipe.MAX_CHAR} karakter",
               CY["600"] if not terlalu_panjang else d.COLOR["critical"], "📏")
        ui.kpi(m3, "Estimasi Proses", f"±{max(len(daftar) * (4 if mode_lengkap else 1), 1)} dtk",
               "perkiraan kasar (setelah model termuat)", d.COLOR["aqua"], "⏱️")
        st.write("")
        input_list = daftar
    else:
        contoh_pilih = st.selectbox("Pilih contoh (opsional, bisa diedit setelahnya)", list(CONTOH.keys()))
        teks_single = st.text_area(
            "Tulis atau tempel komentar di sini",
            value=CONTOH[contoh_pilih], height=120, max_chars=pipe.MAX_CHAR,
            placeholder="Contoh: aplikasinya bagus tapi sering error pas dibuka...",
        )
        st.caption(f"{len(teks_single)} / {pipe.MAX_CHAR} karakter")
        daftar = [teks_single.strip()] if teks_single.strip() else []
        terlalu_panjang = []
        input_list = daftar

    b1, b2 = st.columns([1, 4])
    with b1:
        jalan = st.button("🚀  Jalankan Prediksi", type="primary", use_container_width=True)
    with b2:
        if st.session_state.get("hasil_prediksi"):
            if st.button("🗑️  Bersihkan hasil", use_container_width=False):
                st.session_state.pop("hasil_prediksi", None)
                st.rerun()

    # ---------------- eksekusi ----------------
    if jalan:
        if not input_list:
            st.warning("Tulis dulu komentarnya ya.")
        elif len(input_list) > pipe.MAX_BATCH:
            st.error(f"Jumlah komentar {len(input_list)} melebihi batas {pipe.MAX_BATCH}. Kurangi dulu ya.")
        elif terlalu_panjang:
            st.error(f"Komentar nomor {', '.join(map(str, terlalu_panjang))} melebihi "
                     f"{pipe.MAX_CHAR} karakter. Persingkat dulu ya.")
        else:
            gagal = False
            with st.spinner("Menyiapkan model... unduhan pertama bisa beberapa menit, selanjutnya cepat."):
                try:
                    pipe.siapkan_model(mode_lengkap)
                except Exception as e:
                    st.error("Gagal menyiapkan model — cek koneksi internet, lalu coba lagi."
                             f"\n\nDetail teknis: {e}")
                    gagal = True

            if not gagal:
                status = st.empty()
                try:
                    with st.spinner("Menganalisis komentar, mohon tunggu..."):
                        hasil_semua = analisis_cached(
                            tuple(input_list), mode_lengkap,
                            _progress_cb=lambda m: status.caption(f"⏳ {m}"),
                        )
                    status.empty()
                    st.session_state["hasil_prediksi"] = {
                        "hasil": hasil_semua, "mode_lengkap": mode_lengkap,
                    }
                except Exception as e:
                    status.empty()
                    st.error("Gagal menjalankan model — kemungkinan tokenizer/model dasar belum bisa "
                             f"diunduh dari HuggingFace Hub (perlu internet).\n\nDetail teknis: {e}")

    # ---------------- tampilkan hasil ----------------
    simpan = st.session_state.get("hasil_prediksi")
    if simpan:
        hasil_semua = simpan["hasil"]
        mode_l = simpan["mode_lengkap"]
        st.divider()

        # ---- ringkasan agregat (kalau batch) ----
        if len(hasil_semua) > 1:
            ui.section("Ringkasan Semua Komentar", f"{len(hasil_semua)} komentar dianalisis", CY["600"])

            valid = [h for h in hasil_semua if h["valid"]]
            no_aspect = [h for h in valid if not h["aspek_terdeteksi"]]
            hitung_aspek = {a: sum(1 for h in valid if a in h["aspek_terdeteksi"]) for a in ASPECT_ORDER}
            hitung_sent = {"positif": 0, "negatif": 0, "netral": 0}
            for h in valid:
                for a in h["aspek_terdeteksi"]:
                    hitung_sent[h["sentimen"][a]["label"]] += 1

            k1, k2, k3, k4 = st.columns(4)
            ui.kpi(k1, "Komentar Diproses", f"{len(hasil_semua)}",
                   f"{len(hasil_semua) - len(valid)} dilewati (terlalu pendek)", CY["600"], "💬")
            ui.kpi(k2, "Terdeteksi Aspek", f"{len(valid) - len(no_aspect)}",
                   f"{len(no_aspect)} tanpa aspek (NO_ASPECT)", d.COLOR["aqua"], "🧩")
            ui.kpi(k3, "Pasangan Aspek-Sentimen", f"{sum(hitung_sent.values())}",
                   "satu komentar bisa punya beberapa aspek", d.COLOR["violet"], "🔗")
            dominan = max(hitung_sent, key=hitung_sent.get) if sum(hitung_sent.values()) else "—"
            ui.kpi(k4, "Sentimen Dominan", dominan.capitalize(),
                   f"positif {hitung_sent['positif']} · negatif {hitung_sent['negatif']} · netral {hitung_sent['netral']}",
                   SENT_LABEL_COLOR.get(dominan, (d.COLOR["muted"],))[0], "📊")

            st.write("")
            g1, g2 = st.columns(2)
            with g1:
                df_a = pd.DataFrame({"Aspek": list(hitung_aspek.keys()),
                                     "Jumlah Komentar": list(hitung_aspek.values())})
                fig = px.bar(df_a, x="Aspek", y="Jumlah Komentar", color="Aspek",
                             color_discrete_map=d.ASPECT_COLOR, text="Jumlah Komentar")
                fig.update_traces(textposition="outside", marker_line_width=0)
                fig.update_yaxes(range=[0, max(max(hitung_aspek.values()), 1) * 1.25])
                fig = style_fig(fig, title="Aspek Terdeteksi pada Batch Ini", height=330, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            with g2:
                df_s = pd.DataFrame({
                    "Sentimen": ["Positif", "Netral", "Negatif"],
                    "Jumlah": [hitung_sent["positif"], hitung_sent["netral"], hitung_sent["negatif"]],
                })
                df_s = df_s[df_s["Jumlah"] > 0]  # sembunyikan kelas nol biar label tidak menumpuk
                if len(df_s):
                    fig = px.pie(df_s, names="Sentimen", values="Jumlah", hole=0.58,
                                 color="Sentimen", color_discrete_map=d.SENTIMENT_COLOR)
                    fig.update_traces(textinfo="value+percent",
                                      marker=dict(line=dict(color="#fff", width=2)))
                    fig = style_fig(fig, title="Sebaran Sentimen (per pasangan aspek)", height=330)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    ui.info("Belum ada pasangan aspek-sentimen pada batch ini.", "neutral")

            df_ringkas = pd.DataFrame(pipe.ringkas_untuk_tabel(hasil_semua))
            st.dataframe(
                df_ringkas, hide_index=True, use_container_width=True,
                column_config={
                    "No": st.column_config.NumberColumn("No", width="small"),
                    "Komentar": st.column_config.TextColumn("Komentar", width="medium"),
                    "Aspek Terdeteksi": st.column_config.TextColumn("Aspek Terdeteksi", width="small"),
                    "Sentimen": st.column_config.TextColumn("Sentimen", width="small"),
                    "Sub-Topik": st.column_config.TextColumn("Sub-Topik", width="large"),
                },
            )
            st.download_button(
                "⬇️  Unduh hasil (CSV)",
                df_ringkas.to_csv(index=False, sep=";").encode("utf-8-sig"),
                file_name="hasil_prediksi_komentar.csv", mime="text/csv",
            )
            st.divider()

        # ---- detail per komentar ----
        ui.section("Detail per Komentar",
                   "Buka tab untuk melihat rincian tiap komentar" if len(hasil_semua) > 1 else "",
                   d.COLOR["orange"])

        def render_detail(h, idx):
            st.markdown(
                f'<div class="card card-accent card-soft" style="--accent:{CY["600"]};">'
                f'<div class="card-title">🧹 Hasil Preprocessing</div>'
                f'<div class="card-desc"><b>Teks asli:</b><br>{h["teks_asli"]}<br><br>'
                f'<b>Setelah dibersihkan:</b><br>'
                f'{h["teks_bersih"] if h["teks_bersih"] else "<i>(kosong)</i>"}<br><br>'
                f'<span style="color:{d.COLOR["muted"]};">{h["jumlah_kata"]} kata setelah preprocessing</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

            if not h["valid"]:
                ui.info(
                    f"Teks terlalu pendek (&lt; {pipe.MIN_KATA} kata setelah preprocessing). Di penelitian ini "
                    "ulasan sependek ini otomatis dibuang pada tahap filtering, sehingga model tidak "
                    "dilatih untuk kasus seperti ini.", "danger",
                )
                return

            st.markdown("**🤖 Deteksi Aspek (DistilBERT)** — threshold 0,5 untuk semua aspek")
            ac = st.columns(4)
            for col, a in zip(ac, ASPECT_ORDER):
                p = h["aspek"][a]["probabilitas"]
                aktif = h["aspek"][a]["terdeteksi"]
                warna = d.ASPECT_COLOR[a] if aktif else d.COLOR["muted"]
                status_badge = ui.badge("TERDETEKSI", "#e6f7fb", warna, warna) if aktif else \
                    ui.badge("tidak", d.COLOR["page"], d.COLOR["muted"], d.COLOR["grid"])
                col.markdown(
                    f'<div class="kpi" style="--accent:{warna};">'
                    f'<div class="kpi-top"><span class="kpi-label">{a}</span></div>'
                    f'<div class="kpi-value" style="color:{warna};">{p:.3f}</div>'
                    f'{ui.progress_bar(p, warna)}'
                    f'<div style="margin-top:8px;">{status_badge}</div></div>',
                    unsafe_allow_html=True,
                )

            st.write("")
            fig = px.bar(
                pd.DataFrame({"Aspek": ASPECT_ORDER,
                              "Probabilitas": [h["aspek"][a]["probabilitas"] for a in ASPECT_ORDER]}),
                x="Aspek", y="Probabilitas", color="Aspek", color_discrete_map=d.ASPECT_COLOR,
                text="Probabilitas",
            )
            fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", marker_line_width=0)
            fig.add_hline(y=0.5, line_dash="dot", line_color=d.COLOR["red"], annotation_text="Threshold 0,5")
            fig.update_yaxes(range=[0, 1.16])
            fig = style_fig(fig, title="Probabilitas Aspek (sigmoid)", height=330, showlegend=False)
            st.plotly_chart(fig, use_container_width=True, key=f"aspek_chart_{idx}")

            if not h["aspek_terdeteksi"]:
                ui.info(
                    "<b>NO_ASPECT</b> — tidak ada aspek yang melewati threshold 0,5. Pada data penelitian, "
                    "10.072 dari 59.620 ulasan (16,9%) juga masuk kategori ini.", "warning",
                )
                return

            st.markdown("**💬 Sentimen per Aspek Terdeteksi**")
            for aspek in h["aspek_terdeteksi"]:
                s = h["sentimen"][aspek]
                fg, bg, bd = SENT_LABEL_COLOR.get(s["label"], (d.COLOR["ink"], "#fff", d.COLOR["grid"]))
                warna_aspek = d.ASPECT_COLOR[aspek]

                st.markdown(
                    f'<div class="card card-accent" style="--accent:{warna_aspek};">'
                    f'<div class="card-title"><span style="color:{warna_aspek};">{aspek}</span>'
                    f'{ui.badge(s["label"].upper(), bg, fg, bd)}</div>'
                    f'<div class="card-desc">'
                    f'positif <b>{s["probabilitas"]["positif"]:.2f}</b> · '
                    f'negatif <b>{s["probabilitas"]["negatif"]:.2f}</b> · '
                    f'netral <b>{s["probabilitas"]["netral"]:.2f}</b></div></div>',
                    unsafe_allow_html=True,
                )

                if mode_l:
                    t1, t2 = st.columns(2)
                    with t1:
                        topik = h["topik"].get(aspek)
                        if topik:
                            kata = ", ".join(topik["kata_representatif"])
                            st.markdown(
                                f'<div class="card card-accent" style="--accent:{d.COLOR["aqua"]};">'
                                f'<div class="card-title">🧩 Sub-Topik LDA</div>'
                                f'<div class="card-desc"><b>{topik["nama_subtopik"]}</b><br>'
                                f'<span style="color:{d.COLOR["muted"]};">'
                                f'keyakinan {topik["probabilitas"]:.2f} · kata kunci: {kata}</span>'
                                f'</div></div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            ui.card("🧩 Sub-Topik LDA",
                                    "Tidak dapat ditentukan — kata dalam komentar ini belum cukup "
                                    "dikenali kosakata LDA untuk aspek tersebut.", d.COLOR["muted"])
                    with t2:
                        x = h["xnli_sentimen"].get(aspek)
                        if x:
                            sama = x["label"] == s["label"]
                            tanda = ui.badge("✓ sama", "#eefaee", d.COLOR["good"], "#bfe6bf") if sama else \
                                ui.badge("≠ beda", "#fff8e8", d.COLOR["warning"], "#f5e0b0")
                            det = " · ".join(f"{k} {v:.2f}" for k, v in x["probabilitas"].items())
                            st.markdown(
                                f'<div class="card card-accent" style="--accent:{d.COLOR["orange"]};">'
                                f'<div class="card-title">⚖️ Pembanding Zero-shot (XNLI) {tanda}</div>'
                                f'<div class="card-desc">Sentimen XNLI: <b>{x["label"].upper()}</b><br>'
                                f'<span style="color:{d.COLOR["muted"]};">{det}</span></div></div>',
                                unsafe_allow_html=True,
                            )

            if mode_l and h["xnli_aspek"]:
                st.write("")
                st.markdown("**⚖️ Pembanding Deteksi Aspek: DistilBERT vs Zero-shot XNLI**")
                df_b = pd.DataFrame([
                    {"Aspek": a, "DistilBERT (fine-tuned)": h["aspek"][a]["probabilitas"],
                     "XNLI (zero-shot)": h["xnli_aspek"][a]}
                    for a in ASPECT_ORDER
                ])
                fig = px.bar(df_b.melt(id_vars="Aspek", var_name="Model", value_name="Skor"),
                             x="Aspek", y="Skor", color="Model", barmode="group",
                             color_discrete_sequence=[CY["600"], d.COLOR["orange"]])
                fig.update_traces(marker_line_width=0)
                fig.add_hline(y=0.5, line_dash="dot", line_color=d.COLOR["muted"])
                fig = style_fig(fig, title="Skor Aspek: DistilBERT vs XNLI", height=360)
                st.plotly_chart(fig, use_container_width=True, key=f"banding_chart_{idx}")
                ui.info(
                    "XNLI adalah model <b>zero-shot</b> yang dipakai penelitian ini untuk melabeli data "
                    "latih secara otomatis (tanpa fine-tuning), sedangkan DistilBERT sudah "
                    "<b>di-fine-tune</b> dari label tersebut. Perbedaan hasil keduanya wajar, dan justru "
                    "menjadi salah satu temuan yang dibahas di halaman Performa Model DistilBERT."
                )

        if len(hasil_semua) == 1:
            render_detail(hasil_semua[0], 0)
        else:
            labels = []
            for i, h in enumerate(hasil_semua, start=1):
                if not h["valid"]:
                    tag = "⚠️"
                elif not h["aspek_terdeteksi"]:
                    tag = "○"
                else:
                    labs = {h["sentimen"][a]["label"] for a in h["aspek_terdeteksi"]}
                    tag = "🟢" if labs == {"positif"} else ("🔴" if labs == {"negatif"} else "🟡")
                labels.append(f"{tag} #{i}")
            for i, (tab, h) in enumerate(zip(st.tabs(labels), hasil_semua)):
                with tab:
                    render_detail(h, i)
