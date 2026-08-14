"""
Semua chart statis (yang datanya tidak berubah) dibangun di sini dan di-cache dengan
`st.cache_resource`, sehingga hanya dibuat SEKALI selama aplikasi berjalan. Berpindah
halaman tidak lagi membangun ulang puluhan figure Plotly.

Catatan: figure hasil cache TIDAK boleh dimutasi di luar modul ini (jangan panggil
fig.update_*), karena objeknya dipakai bersama antar rerun.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import data as d
import ui

CY = ui.CYAN
ASPECT_ORDER = ["Individual", "Technical", "Social", "Financial"]
SENTIMENT_ORDER = ["Positif", "Netral", "Negatif"]

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=d.COLOR["ink"], size=12),
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
    margin=dict(l=10, r=10, t=46, b=60),
    legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0),
    # text="" penting: tanpa ini Plotly menampilkan teks "undefined" pada chart tanpa judul
    title=dict(text="", font=dict(size=14, color=d.COLOR["ink"])),
    hoverlabel=dict(bgcolor="#ffffff", bordercolor=d.COLOR["grid"], font_size=12),
)


def style_fig(fig, **kwargs):
    layout = dict(PLOTLY_LAYOUT)
    layout.update(kwargs)
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor=d.COLOR["grid"], zerolinecolor=d.COLOR["grid"])
    fig.update_yaxes(gridcolor=d.COLOR["grid"], zerolinecolor=d.COLOR["grid"])
    return fig


def rb(n):
    """Format angka gaya Indonesia: 59620 -> 59.620 (pemisah ribuan titik)."""
    return f"{n:,}".replace(",", ".")


# ===========================================================================
# DASHBOARD UTAMA
# ===========================================================================
@st.cache_resource(show_spinner=False)
def dashboard():
    f = {}

    df = d.df_aspect_distribution.sort_values("Jumlah Ulasan")
    fig = px.bar(df, x="Jumlah Ulasan", y="Aspek", orientation="h",
                 color="Aspek", color_discrete_map=d.ASPECT_COLOR, text="% dari Terlabel")
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside", marker_line_width=0)
    fig.update_xaxes(range=[0, df["Jumlah Ulasan"].max() * 1.2])
    f["aspek_dist"] = style_fig(fig, title="Distribusi Aspek Terlabel (multi-label, total bisa >100%)",
                                height=360, showlegend=False)

    fig = px.bar(d.df_active_aspects, x="Jumlah Aspek Aktif", y="Jumlah Ulasan",
                 text="Jumlah Ulasan", color="Jumlah Ulasan",
                 color_continuous_scale=[CY["200"], CY["700"]])
    fig.update_traces(texttemplate="%{text:,}", textposition="outside", marker_line_width=0)
    f["aspek_aktif"] = style_fig(fig, title="Jumlah Aspek Aktif per Ulasan", height=360,
                                 coloraxis_showscale=False)

    fig = px.pie(d.df_sentiment_distribution, names="Kelas Sentimen", values="Jumlah Pasangan",
                 hole=0.58, color="Kelas Sentimen", color_discrete_map=d.SENTIMENT_COLOR)
    fig.update_traces(textinfo="percent", marker=dict(line=dict(color="#ffffff", width=2)))
    f["sent_pie"] = style_fig(fig, title="Sentimen Keseluruhan", height=360)

    cross = d.df_aspect_sentiment_cross[["Aspek", "Negatif", "Netral", "Positif"]].melt(
        id_vars="Aspek", var_name="Sentimen", value_name="Jumlah")
    fig = px.bar(cross, x="Aspek", y="Jumlah", color="Sentimen", barmode="stack",
                 color_discrete_map=d.SENTIMENT_COLOR,
                 category_orders={"Aspek": ASPECT_ORDER, "Sentimen": ["Positif", "Netral", "Negatif"]})
    fig.update_traces(marker_line=dict(color="#ffffff", width=1.5))
    f["sent_stack"] = style_fig(fig, title="Sentimen per Aspek", height=360)

    fig = px.line(d.df_ratio_aspect, x="Rasio",
                  y=["Akurasi Val", "F1 Weighted (Gold)", "F1 Macro (Gold)"], markers=True,
                  color_discrete_sequence=[CY["500"], d.COLOR["aqua"], d.COLOR["orange"]])
    fig.update_traces(line=dict(width=2.5), marker=dict(size=9))
    f["ratio_aspek"] = style_fig(fig, title="Performa Model Aspek per Rasio Split", height=340,
                                 yaxis_title="Skor")

    plot_df = d.df_aspect_gold_report[d.df_aspect_gold_report["Aspek"].isin(ASPECT_ORDER)]
    fig = px.bar(plot_df, x="Aspek", y="F1-Score", color="Aspek",
                 color_discrete_map=d.ASPECT_COLOR, text="F1-Score")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", marker_line_width=0)
    fig.update_yaxes(range=[0, 1.05])
    f["f1_aspek"] = style_fig(fig, title="F1-Score per Aspek (vs Gold)", height=340, showlegend=False)

    fig = px.line(d.df_ratio_sentiment, x="Rasio",
                  y=["Akurasi Val", "F1 Weighted (Gold)", "F1 Macro (Gold)"], markers=True,
                  color_discrete_sequence=[CY["500"], d.COLOR["aqua"], d.COLOR["orange"]])
    fig.update_traces(line=dict(width=2.5), marker=dict(size=9))
    f["ratio_sent"] = style_fig(fig, title="Performa Model Sentimen per Rasio Split", height=340,
                                yaxis_title="Skor")

    plot_df = d.df_sentiment_gold_report[d.df_sentiment_gold_report["Kelas"].isin(SENTIMENT_ORDER)]
    fig = px.bar(plot_df, x="Kelas", y="F1-Score", color="Kelas",
                 color_discrete_map=d.SENTIMENT_COLOR, text="F1-Score",
                 category_orders={"Kelas": SENTIMENT_ORDER})
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", marker_line_width=0)
    fig.update_yaxes(range=[0, 1.05])
    f["f1_sent"] = style_fig(fig, title="F1-Score per Kelas Sentimen (vs Gold)", height=340,
                             showlegend=False)
    return f


# ===========================================================================
# DETAIL 1 — DATA & PREPROCESSING
# ===========================================================================
@st.cache_resource(show_spinner=False)
def data_prep():
    f = {}
    fig = px.bar(d.df_apps.sort_values("Jumlah Ulasan"), x="Jumlah Ulasan", y="Aplikasi",
                 orientation="h", text="Persentase",
                 color="Jumlah Ulasan", color_continuous_scale=[CY["200"], CY["700"]])
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside", marker_line_width=0)
    fig.update_xaxes(range=[0, d.df_apps["Jumlah Ulasan"].max() * 1.16])
    f["apps"] = style_fig(fig, title="Jumlah Ulasan per Aplikasi (Total 100.109)", height=430,
                          xaxis_title="Jumlah Ulasan", yaxis_title="", coloraxis_showscale=False)

    fig = go.Figure(go.Funnel(
        y=["Hasil Web Scraping", "Hapus Duplikat (-30.629)", "Hapus Simbol/No-text (-821)",
           "Data Preparation Selesai"],
        x=[100109, 100109 - 30629, 100109 - 30629 - 821, 68659],
        marker=dict(color=[CY["700"], CY["600"], CY["500"], CY["400"]]),
        textinfo="value+percent initial",
        connector=dict(line=dict(color=d.COLOR["grid"], width=1)),
    ))
    f["funnel_prep"] = style_fig(fig, title="Data Preparation: 100.109 → 68.659", height=380)

    fig = go.Figure(go.Funnel(
        y=["Sebelum Penyaringan", "Hapus Ulasan Kosong", "Hapus Ulasan < 3 Kata (-9.039)"],
        x=[68659, 68659, 59620],
        marker=dict(color=[d.COLOR["aqua"], d.COLOR["aqua"], d.COLOR["good"]]),
        textinfo="value+percent initial",
        connector=dict(line=dict(color=d.COLOR["grid"], width=1)),
    ))
    f["funnel_filter"] = style_fig(fig, title="Data Filtering: 68.659 → 59.620", height=380)
    return f


# ===========================================================================
# DETAIL 2 — GOLD STANDARD
# ===========================================================================
@st.cache_resource(show_spinner=False)
def gold():
    f = {}
    fig = px.pie(d.df_sampling, names="Kelompok Sampling", values="Jumlah", hole=0.58,
                 color_discrete_sequence=[CY["600"], d.COLOR["yellow"], d.COLOR["aqua"]])
    fig.update_traces(textinfo="value+percent", marker=dict(line=dict(color="#fff", width=2)))
    f["sampling"] = style_fig(fig, title="Komposisi Sampling (n=500)", height=340)

    fig = px.bar(d.df_gold_aspect_support, x="Aspek", y="Support (label=1)",
                 color="Aspek", color_discrete_map=d.ASPECT_COLOR, text="Support (label=1)")
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_yaxes(range=[0, d.df_gold_aspect_support["Support (label=1)"].max() * 1.18])
    f["support"] = style_fig(fig, title="Support Label Aspek", height=340, showlegend=False)

    fig = px.pie(d.df_gold_sentiment, names="Kelas", values="Jumlah", hole=0.58,
                 color="Kelas", color_discrete_map=d.SENTIMENT_COLOR)
    fig.update_traces(textinfo="value+percent", marker=dict(line=dict(color="#fff", width=2)))
    f["gold_sent"] = style_fig(fig, title="Distribusi Sentimen Gold (n=378)", height=340)

    fig = px.bar(d.df_kappa_aspect, x="Aspek", y="Nilai Kappa", color="Aspek",
                 color_discrete_map={**d.ASPECT_COLOR, "Rata-rata": d.COLOR["violet"]},
                 text="Interpretasi")
    fig.add_hline(y=d.KAPPA_SENTIMENT["nilai"], line_dash="dot", line_color=d.COLOR["red"],
                  annotation_text=f"Kappa Sentimen = {d.KAPPA_SENTIMENT['nilai']}",
                  annotation_position="top left")
    fig.update_traces(textposition="outside", marker_line_width=0)
    f["kappa"] = style_fig(fig, title="Cohen's Kappa per Aspek vs Kappa Sentimen", height=400,
                           yaxis_title="Nilai Kappa", showlegend=False)

    auc_long = d.df_auc_scheme.melt(id_vars="Aspek", var_name="Skema", value_name="AUC")
    fig = px.bar(auc_long, x="Aspek", y="AUC", color="Skema", barmode="group",
                 color_discrete_sequence=[CY["600"], d.COLOR["orange"]])
    fig.update_traces(marker_line_width=0)
    f["auc"] = style_fig(fig, title="AUC Skema A (Bawaan) vs Skema B (Custom)", height=380)

    fig = px.bar(d.df_threshold_aspect, x="Aspek", y="Threshold Optimal", color="Aspek",
                 color_discrete_map=d.ASPECT_COLOR, text="F1 (saat pencarian)")
    fig.update_traces(texttemplate="F1=%{text:.4f}", textposition="outside", marker_line_width=0)
    f["threshold"] = style_fig(fig, title="Threshold Optimal per Aspek (Skema B)", height=380,
                               showlegend=False)
    return f


# ===========================================================================
# DETAIL 3 — PELABELAN OTOMATIS (XNLI)
# ===========================================================================
@st.cache_resource(show_spinner=False)
def xnli():
    f = {}
    fig = px.bar(d.df_active_aspects, x="Jumlah Aspek Aktif", y="Jumlah Ulasan",
                 text="Jumlah Ulasan", color="Jumlah Ulasan",
                 color_continuous_scale=[CY["200"], CY["700"]])
    fig.update_traces(texttemplate="%{text:,}", textposition="outside", marker_line_width=0)
    f["aktif"] = style_fig(fig, title="Jumlah Aspek Aktif per Ulasan (Multi-label)", height=380,
                           coloraxis_showscale=False)

    df = d.df_aspect_distribution.sort_values("Jumlah Ulasan")
    fig = px.bar(df, x="Jumlah Ulasan", y="Aspek", orientation="h",
                 color="Aspek", color_discrete_map=d.ASPECT_COLOR, text="% dari Terlabel")
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside", marker_line_width=0)
    fig.update_xaxes(range=[0, df["Jumlah Ulasan"].max() * 1.2])
    f["aspek_dist"] = style_fig(fig, title="Distribusi Aspek Terlabel", height=380, showlegend=False)

    fig = px.pie(d.df_sentiment_distribution, names="Kelas Sentimen", values="Jumlah Pasangan",
                 hole=0.58, color="Kelas Sentimen", color_discrete_map=d.SENTIMENT_COLOR)
    fig.update_traces(textinfo="value+percent", marker=dict(line=dict(color="#fff", width=2)))
    f["sent_pie"] = style_fig(fig, title="Distribusi Sentimen Keseluruhan", height=380)

    cross = d.df_aspect_sentiment_cross[["Aspek", "Negatif", "Netral", "Positif"]].melt(
        id_vars="Aspek", var_name="Sentimen", value_name="Jumlah")
    fig = px.bar(cross, x="Aspek", y="Jumlah", color="Sentimen", barmode="stack",
                 color_discrete_map=d.SENTIMENT_COLOR,
                 category_orders={"Aspek": ASPECT_ORDER, "Sentimen": ["Positif", "Netral", "Negatif"]})
    fig.update_traces(marker_line=dict(color="#ffffff", width=1.5))
    f["sent_stack"] = style_fig(fig, title="Distribusi Sentimen per Aspek (stacked)", height=380)

    fig = px.bar(d.df_gold_match_aspect, x="Aspek", y=["F1-Score", "Kappa#2"], barmode="group",
                 color_discrete_sequence=[CY["600"], d.COLOR["violet"]])
    fig.update_traces(marker_line_width=0)
    f["gold_match"] = style_fig(fig, title="F1-Score & Kappa#2 XNLI vs Gold (per Aspek)", height=360)

    fig = px.bar(d.df_sentiment_per_aspect_gold, x="Aspek", y=["F1 Macro", "F1 Weighted", "Kappa"],
                 barmode="group", color_discrete_sequence=[CY["600"], d.COLOR["aqua"], d.COLOR["violet"]])
    fig.update_traces(marker_line_width=0)
    f["sent_per_aspek"] = style_fig(fig, title="Kesesuaian Sentimen dengan Gold per Aspek", height=360)
    return f


# ===========================================================================
# DETAIL 4 — PERFORMA MODEL
# ===========================================================================
@st.cache_resource(show_spinner=False)
def performa():
    f = {}
    fig = px.line(d.df_ratio_aspect, x="Rasio",
                  y=["Akurasi Train", "Akurasi Val", "F1 Weighted (Gold)", "F1 Macro (Gold)"],
                  markers=True,
                  color_discrete_sequence=[d.COLOR["muted"], CY["500"], d.COLOR["aqua"], d.COLOR["orange"]])
    fig.update_traces(line=dict(width=2.5), marker=dict(size=9))
    f["ratio_aspek"] = style_fig(fig, title="Performa Model Aspek per Rasio Split", height=400,
                                 yaxis_title="Skor")

    fig = px.line(d.df_ratio_sentiment, x="Rasio",
                  y=["Akurasi Train", "Akurasi Val", "F1 Weighted (Gold)", "F1 Macro (Gold)"],
                  markers=True,
                  color_discrete_sequence=[d.COLOR["muted"], CY["500"], d.COLOR["aqua"], d.COLOR["orange"]])
    fig.update_traces(line=dict(width=2.5), marker=dict(size=9))
    f["ratio_sent"] = style_fig(fig, title="Performa Model Sentimen per Rasio Split", height=400,
                                yaxis_title="Skor")

    plot_df = d.df_aspect_gold_report[d.df_aspect_gold_report["Aspek"].isin(ASPECT_ORDER)]
    fig = px.bar(plot_df, x="Aspek", y=["Precision", "Recall", "F1-Score"], barmode="group",
                 color_discrete_sequence=[CY["600"], d.COLOR["orange"], d.COLOR["aqua"]])
    fig.update_traces(marker_line_width=0)
    f["aspek_prf"] = style_fig(fig, title="Precision / Recall / F1 per Aspek (vs Gold)", height=380)

    fig = px.bar(d.df_threshold_tuning, x="Aspek", y="Delta", color="Aspek",
                 color_discrete_map=d.ASPECT_COLOR, text="Delta")
    fig.update_traces(texttemplate="%{text:+.4f}", textposition="outside", marker_line_width=0)
    fig.add_hline(y=0, line_color=d.COLOR["muted"])
    f["threshold_delta"] = style_fig(fig, title="Delta F1-Score Setelah Threshold Tuning", height=320,
                                     showlegend=False)

    plot_df = d.df_sentiment_gold_report[d.df_sentiment_gold_report["Kelas"].isin(SENTIMENT_ORDER)]
    fig = px.bar(plot_df, x="Kelas", y=["Precision", "Recall", "F1-Score"], barmode="group",
                 color_discrete_sequence=[CY["600"], d.COLOR["orange"], d.COLOR["aqua"]],
                 category_orders={"Kelas": SENTIMENT_ORDER})
    fig.update_traces(marker_line_width=0)
    f["sent_prf"] = style_fig(fig, title="Precision / Recall / F1 per Kelas Sentimen (vs Gold)",
                              height=380)

    fig = px.pie(d.df_netral_error, names="Prediksi", values="Jumlah", hole=0.58,
                 color_discrete_sequence=[d.COLOR["good"], d.COLOR["critical"], d.COLOR["warning"]])
    fig.update_traces(textinfo="value+percent", marker=dict(line=dict(color="#fff", width=2)))
    f["netral_err"] = style_fig(fig, title="Prediksi Model untuk Data Bergold Netral (n=30)", height=340)
    return f


# ===========================================================================
# DETAIL 5 — TOPIC MODELING
# ===========================================================================
@st.cache_resource(show_spinner=False)
def lda():
    f = {}
    fig = px.bar(d.df_lda_corpus, x="Aspek",
                 y=["Ukuran Subset Awal", "Ukuran Korpus Setelah Pra-pemrosesan"],
                 barmode="group", color_discrete_sequence=[d.COLOR["muted"], CY["600"]])
    fig.update_traces(marker_line_width=0)
    f["corpus"] = style_fig(fig, title="Ukuran Korpus per Aspek (Sebelum & Sesudah Pra-pemrosesan)",
                            height=380)

    fig = px.bar(d.df_lda_coherence, x="Aspek", y="Coherence (K Final)", color="Aspek",
                 color_discrete_map=d.ASPECT_COLOR, text="K Final")
    fig.update_traces(texttemplate="K=%{text}", textposition="outside", marker_line_width=0)
    f["coherence"] = style_fig(fig, title="Coherence Score (c_v) pada K Final per Aspek", height=380,
                               showlegend=False)

    fig = px.treemap(d.df_subtopics, path=["Aspek", "Sub-Topik"], values="Jumlah Dokumen",
                     color="Aspek", color_discrete_map=d.ASPECT_COLOR, custom_data=["% dari Aspek"])
    fig.update_traces(
        texttemplate="<b>%{label}</b><br>%{value:,} dok (%{customdata[0]:.1f}%)",
        textfont_size=11, marker=dict(line=dict(color="#ffffff", width=2)),
        pathbar=dict(visible=False), root_color="#ffffff",
        hovertemplate="<b>%{label}</b><br>%{value:,} dokumen<extra></extra>",
    )
    f["treemap"] = style_fig(fig, height=560, margin=dict(l=4, r=4, t=10, b=4))
    return f


# ===========================================================================
# DETAIL 6 — VALIDASI AHLI
# ===========================================================================
@st.cache_resource(show_spinner=False)
def validasi():
    f = {}
    stmt = d.df_validation_statements.copy()
    stmt["Label"] = stmt["No"].astype(str) + ". " + stmt["Pernyataan"].str.slice(0, 45) + "..."
    fig = px.bar(stmt, x="Rata-rata", y="Label", orientation="h",
                 color="Rata-rata", color_continuous_scale=[CY["200"], CY["700"]], text="Rata-rata")
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside", marker_line_width=0)
    fig.update_xaxes(range=[0, 5.4])
    f["stmt"] = style_fig(fig, title="Skor Rata-rata per Pernyataan", height=340, yaxis_title="",
                          coloraxis_showscale=False)

    fig = px.bar(d.df_validation_subtopic, x="Aspek",
                 y=["Relevansi Rata-Rata", "Kejelasan Rata-Rata"], barmode="group",
                 color_discrete_sequence=[CY["600"], d.COLOR["aqua"]],
                 category_orders={"Aspek": ASPECT_ORDER})
    fig.update_traces(marker_line_width=0)
    fig.update_yaxes(range=[0, 5])
    f["subtopic"] = style_fig(fig, title="Skor Relevansi & Kejelasan (rata-rata 2 validator)", height=380)
    return f


# ===========================================================================
# DETAIL 7 — KESIMPULAN
# ===========================================================================
@st.cache_resource(show_spinner=False)
def kesimpulan():
    f = {}
    fig = px.bar(d.df_gap_test_gold, x="Aspek", y=["F1 Data Test", "F1 Gold"], barmode="group",
                 color_discrete_sequence=[CY["600"], d.COLOR["red"]],
                 category_orders={"Aspek": ASPECT_ORDER})
    fig.update_traces(marker_line_width=0)
    f["gap"] = style_fig(fig, title="F1-Score: Data Test (XNLI) vs Gold Standard", height=380)

    fig = px.scatter(d.df_correlation, x="F1 Gold (Klasifikasi)", y="Coherence Score (K Final)",
                     size="Jumlah Sub-Topik (LDA)", color="Aspek",
                     color_discrete_map=d.ASPECT_COLOR, text="Aspek", size_max=45)
    fig.update_traces(textposition="top center", marker=dict(line=dict(color="#fff", width=2)))
    f["korelasi"] = style_fig(fig, title="F1 Klasifikasi vs Coherence (ukuran = jumlah sub-topik)",
                              height=380)
    return f


# ===========================================================================
# TABEL — Styler juga di-cache (pandas Styler cukup mahal untuk dibangun ulang)
# ===========================================================================
@st.cache_resource(show_spinner=False)
def tables():
    t = {}
    t["apps"] = d.df_apps.style.format({"Jumlah Ulasan": "{:,.0f}", "Persentase": "{:.2f}%"})
    t["cross"] = d.df_aspect_sentiment_cross.style.format({
        "Negatif": "{:,.0f}", "Netral": "{:,.0f}", "Positif": "{:,.0f}", "Total": "{:,.0f}",
        "Negatif %": "{:.1f}%", "Netral %": "{:.1f}%", "Positif %": "{:.1f}%"})
    t["sent_murni"] = d.df_class_report_sentimen_murni_xnli.style.format(
        {"Precision": "{:.2f}", "Recall": "{:.2f}", "F1-Score": "{:.2f}", "Support": "{:.0f}"}, na_rep="")
    t["sent_e2e"] = d.df_class_report_sentimen_e2e_xnli.style.format(
        {"Precision": "{:.2f}", "Recall": "{:.2f}", "F1-Score": "{:.2f}", "Support": "{:.0f}"}, na_rep="")
    t["aspek_gold"] = d.df_aspect_gold_report.style.format(
        {"Precision": "{:.3f}", "Recall": "{:.3f}", "F1-Score": "{:.3f}", "Support": "{:.0f}"})
    t["sent_gold"] = d.df_sentiment_gold_report.style.format(
        {"Precision": "{:.2f}", "Recall": "{:.2f}", "F1-Score": "{:.2f}", "Support": "{:.0f}"}, na_rep="")
    t["sent_test"] = d.df_sentiment_test_report.style.format(
        {"Precision": "{:.2f}", "Recall": "{:.2f}", "F1-Score": "{:.2f}", "Support": "{:,.0f}"}, na_rep="")
    t["ratio_summary"] = d.df_ratio_summary.style.format({
        "F1 Weighted Terbaik": "{:.4f}", "F1 Weighted Kedua": "{:.4f}", "Selisih Weighted": "{:+.4f}",
        "F1 Macro Terbaik": "{:.4f}", "F1 Macro Kedua": "{:.4f}", "Selisih Macro": "{:+.4f}"})
    t["gap"] = d.df_gap_test_gold.style.format(
        {"F1 Data Test": "{:.3f}", "F1 Gold": "{:.3f}", "Selisih": "{:+.3f}"})
    corr_display = d.df_correlation.rename(columns={
        "F1 Gold (Klasifikasi)": "F1 Gold", "Jumlah Sub-Topik (LDA)": "Sub-Topik",
        "Coherence Score (K Final)": "Coherence"})
    t["korelasi"] = corr_display.style.format({"F1 Gold": "{:.3f}", "Coherence": "{:.4f}"})
    return t


@st.cache_resource(show_spinner=False)
def subtopik_table(aspek_filter):
    df = d.df_subtopics if aspek_filter == "Semua" else d.df_subtopics[d.df_subtopics["Aspek"] == aspek_filter]
    return df.style.format({"Jumlah Dokumen": "{:,.0f}", "% dari Aspek": "{:.2f}%"})
