"""
Komponen UI reusable + sistem styling dashboard.
Semua warna mengacu ke token di data.py (aksen utama: cyan).
"""
import streamlit as st

import data as d

# ---------------------------------------------------------------------------
# Token warna turunan (ramp cyan sebagai aksen utama)
# ---------------------------------------------------------------------------
CYAN = {
    "50": "#ecfeff",
    "100": "#cffafe",
    "200": "#a5f3fc",
    "300": "#67e8f9",
    "400": "#22d3ee",
    "500": "#06b6d4",
    "600": "#0891b2",
    "700": "#0e7490",
    "800": "#155e75",
    "900": "#164e63",
}

GRAD_MAIN = f"linear-gradient(135deg, {CYAN['700']} 0%, {CYAN['500']} 55%, {CYAN['300']} 100%)"
GRAD_SOFT = f"linear-gradient(135deg, {CYAN['50']} 0%, #ffffff 100%)"


def inject_css():
    st.markdown(
        f"""
<style>
  /* ---------- layout dasar ---------- */
  .block-container {{ padding-top: 1.4rem; padding-bottom: 3.5rem; max-width: 1440px; }}
  footer, [data-testid="stMainMenu"], [data-testid="stAppDeployButton"] {{ visibility: hidden; }}
  header {{ background: transparent; }}
  [data-testid="stAppViewContainer"] {{ background: {d.COLOR['page']}; }}

  /* ---------- POPUP LOADING (spinner jadi modal tengah layar + scrim) ---------- */
  [data-testid="stSpinner"] {{
      position: fixed !important;
      top: 50%; left: 50%;
      transform: translate(-50%, -50%);
      z-index: 99999;
      background: #ffffff !important;
      padding: 26px 32px;
      border-radius: 16px;
      border: 1px solid {CYAN['100']};
      border-top: 4px solid {CYAN['500']};
      min-width: 300px;
      max-width: 470px;
      /* shadow kedua = scrim gelap seluruh layar tanpa elemen tambahan */
      box-shadow: 0 22px 60px rgba(6, 42, 52, 0.35),
                  0 0 0 100vmax rgba(8, 32, 40, 0.45);
  }}
  [data-testid="stSpinner"] p {{
      color: {d.COLOR['ink']} !important;
      font-weight: 600;
      font-size: 0.95rem;
      margin: 0;
  }}
  [data-testid="stSpinnerIcon"] {{
      border-top-color: {CYAN['500']} !important;
      border-right-color: {CYAN['500']} !important;
      width: 1.5rem !important; height: 1.5rem !important;
  }}

  /* ---------- HERO ---------- */
  .hero {{
      background: {GRAD_MAIN};
      border-radius: 18px;
      padding: 26px 30px;
      color: #ffffff;
      margin-bottom: 20px;
      box-shadow: 0 10px 30px rgba(14, 116, 144, 0.25);
  }}
  .hero-kicker {{
      text-transform: uppercase; letter-spacing: 0.09em;
      font-size: 0.72rem; font-weight: 700; opacity: 0.85; margin-bottom: 6px;
  }}
  .hero-title {{ font-size: 1.85rem; font-weight: 800; line-height: 1.2; margin-bottom: 6px; }}
  .hero-sub {{ font-size: 0.98rem; opacity: 0.94; line-height: 1.5; max-width: 900px; }}
  .hero-meta {{
      margin-top: 14px; display: flex; flex-wrap: wrap; gap: 8px;
  }}
  .hero-chip {{
      background: rgba(255,255,255,0.18);
      border: 1px solid rgba(255,255,255,0.32);
      border-radius: 999px; padding: 4px 13px;
      font-size: 0.78rem; font-weight: 600;
  }}

  /* ---------- KPI CARD ---------- */
  .kpi {{
      background: #ffffff;
      border: 1px solid {d.COLOR['grid']};
      border-radius: 14px;
      padding: 15px 17px 14px 17px;
      height: 100%;
      position: relative;
      overflow: hidden;
      transition: transform .16s ease, box-shadow .16s ease;
  }}
  .kpi:hover {{ transform: translateY(-3px); box-shadow: 0 12px 26px rgba(11,11,11,0.09); }}
  .kpi::before {{
      content: ""; position: absolute; top: 0; left: 0; right: 0; height: 4px;
      background: var(--accent, {CYAN['500']});
  }}
  .kpi-top {{ display: flex; align-items: center; gap: 7px; margin-bottom: 7px; }}
  .kpi-icon {{ font-size: 1rem; line-height: 1; }}
  .kpi-label {{
      color: {d.COLOR['ink2']}; font-size: 0.74rem; font-weight: 700;
      text-transform: uppercase; letter-spacing: 0.045em;
  }}
  .kpi-value {{
      color: {d.COLOR['ink']}; font-size: 1.72rem; font-weight: 800; line-height: 1.1;
  }}
  .kpi-sub {{ color: {d.COLOR['muted']}; font-size: 0.76rem; margin-top: 4px; line-height: 1.35; }}

  /* ---------- CARD UMUM ---------- */
  .card {{
      background: #ffffff;
      border: 1px solid {d.COLOR['grid']};
      border-radius: 14px;
      padding: 16px 18px;
      margin-bottom: 12px;
      transition: box-shadow .16s ease;
  }}
  .card:hover {{ box-shadow: 0 8px 22px rgba(11,11,11,0.07); }}
  .card-accent {{ border-left: 4px solid var(--accent, {CYAN['500']}); }}
  .card-title {{
      font-weight: 700; font-size: 0.98rem; color: {d.COLOR['ink']};
      display: flex; align-items: center; gap: 8px; margin-bottom: 4px;
  }}
  .card-desc {{ color: {d.COLOR['ink2']}; font-size: 0.86rem; line-height: 1.5; }}
  .card-soft {{ background: {GRAD_SOFT}; border-color: {CYAN['100']}; }}
  .card-eq {{ min-height: 148px; }}

  /* ---------- NAV CARD (dashboard utama) ---------- */
  .navcard {{
      background: #ffffff; border: 1px solid {d.COLOR['grid']};
      border-radius: 14px; padding: 18px; height: 100%;
      border-top: 4px solid var(--accent, {CYAN['500']});
      transition: transform .16s ease, box-shadow .16s ease;
  }}
  .navcard:hover {{ transform: translateY(-3px); box-shadow: 0 14px 30px rgba(11,11,11,0.10); }}
  .navcard-icon {{ font-size: 1.5rem; margin-bottom: 8px; }}
  .navcard-title {{ font-weight: 800; font-size: 1.02rem; color: {d.COLOR['ink']}; margin-bottom: 5px; }}
  .navcard-desc {{ color: {d.COLOR['ink2']}; font-size: 0.85rem; line-height: 1.45; }}

  /* ---------- SECTION HEADING ---------- */
  .sec {{ margin: 0.4rem 0 0.1rem 0; display: flex; align-items: center; gap: 9px; }}
  .sec-bar {{ width: 4px; height: 21px; border-radius: 3px; background: var(--accent, {CYAN['500']}); }}
  .sec-title {{ font-size: 1.28rem; font-weight: 800; color: {d.COLOR['ink']}; }}
  .sec-sub {{ color: {d.COLOR['ink2']}; font-size: 0.9rem; margin: 2px 0 0.9rem 13px; line-height: 1.45; }}

  /* ---------- INFO BOX ---------- */
  .ibox {{
      border-radius: 10px; padding: 12px 16px; font-size: 0.89rem;
      line-height: 1.55; margin-bottom: 1rem;
      border-left: 4px solid var(--accent, {CYAN['600']});
      background: var(--bg, {CYAN['50']});
      color: {d.COLOR['ink2']};
  }}

  /* ---------- BADGE / PILL ---------- */
  .badge {{
      display: inline-block; border-radius: 999px; padding: 3px 11px;
      font-size: 0.75rem; font-weight: 700; letter-spacing: 0.01em;
      background: var(--bg, {CYAN['100']}); color: var(--fg, {CYAN['800']});
      border: 1px solid var(--bd, {CYAN['200']});
  }}

  /* ---------- SIDEBAR ---------- */
  [data-testid="stSidebar"] {{ background: #ffffff; border-right: 1px solid {d.COLOR['grid']}; }}
  [data-testid="stSidebar"] .stRadio > div {{ gap: 3px; }}
  [data-testid="stSidebar"] .stRadio label {{
      border-radius: 9px; padding: 7px 10px; transition: background .13s ease;
  }}
  [data-testid="stSidebar"] .stRadio label:hover {{ background: {CYAN['50']}; }}
  .side-brand {{
      background: {GRAD_MAIN}; border-radius: 13px; padding: 14px 15px;
      color: #fff; margin-bottom: 14px;
  }}
  .side-brand-title {{ font-weight: 800; font-size: 1rem; line-height: 1.25; }}
  .side-brand-sub {{ font-size: 0.75rem; opacity: 0.92; margin-top: 3px; line-height: 1.35; }}
  .side-label {{
      font-size: 0.7rem; font-weight: 800; letter-spacing: 0.08em;
      text-transform: uppercase; color: {CYAN['700']}; margin: 12px 0 5px 2px;
  }}
  .side-foot {{
      font-size: 0.76rem; color: {d.COLOR['ink2']}; line-height: 1.5;
      background: {d.COLOR['page']}; border: 1px solid {d.COLOR['grid']};
      border-radius: 10px; padding: 11px 13px;
  }}

  /* ---------- TABS ---------- */
  .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
  .stTabs [data-baseweb="tab"] {{
      border-radius: 9px 9px 0 0; padding: 7px 15px; font-weight: 600;
  }}
  .stTabs [aria-selected="true"] {{ background: {CYAN['50']}; color: {CYAN['800']} !important; }}

  /* ---------- MISC ---------- */
  div[data-testid="stMetricValue"] {{ font-size: 1.45rem; font-weight: 800; }}
  .stButton > button {{ border-radius: 10px; font-weight: 700; }}
  .aspect-strip {{ display: flex; gap: 6px; flex-wrap: wrap; margin-top: 8px; }}
  .prog-wrap {{ background: {d.COLOR['grid']}; border-radius: 999px; height: 8px; overflow: hidden; margin-top: 6px; }}
  .prog-fill {{ height: 100%; border-radius: 999px; background: var(--accent, {CYAN['500']}); }}
</style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Komponen
# ---------------------------------------------------------------------------
def _html(markup, target=None):
    """Render HTML tanpa indentasi (indentasi >=4 spasi akan dianggap code block oleh markdown)."""
    (target or st).markdown(markup, unsafe_allow_html=True)


def hero(title, subtitle="", kicker="", chips=None):
    parts = ['<div class="hero">']
    if kicker:
        parts.append(f'<div class="hero-kicker">{kicker}</div>')
    parts.append(f'<div class="hero-title">{title}</div>')
    if subtitle:
        parts.append(f'<div class="hero-sub">{subtitle}</div>')
    if chips:
        parts.append('<div class="hero-meta">')
        parts += [f'<span class="hero-chip">{c}</span>' for c in chips]
        parts.append("</div>")
    parts.append("</div>")
    _html("".join(parts))


def kpi(col, label, value, sub="", accent=None, icon=""):
    accent = accent or CYAN["500"]
    parts = [f'<div class="kpi" style="--accent:{accent};"><div class="kpi-top">']
    if icon:
        parts.append(f'<span class="kpi-icon">{icon}</span>')
    parts.append(f'<span class="kpi-label">{label}</span></div>')
    parts.append(f'<div class="kpi-value">{value}</div>')
    if sub:
        parts.append(f'<div class="kpi-sub">{sub}</div>')
    parts.append("</div>")
    _html("".join(parts), col)


def kpi_custom(col, label, value, extra_html="", accent=None):
    """KPI card dengan blok HTML tambahan di bawah nilai (progress bar, badge, dll)."""
    accent = accent or CYAN["500"]
    _html(
        f'<div class="kpi" style="--accent:{accent};">'
        f'<div class="kpi-top"><span class="kpi-label">{label}</span></div>'
        f'<div class="kpi-value" style="color:{accent};">{value}</div>'
        f'{extra_html}</div>',
        col,
    )


def card(title, desc, accent=None, icon="", soft=False, target=None, equal_height=False):
    accent = accent or CYAN["500"]
    cls = "card card-accent" + (" card-soft" if soft else "") + (" card-eq" if equal_height else "")
    _html(
        f'<div class="{cls}" style="--accent:{accent};">'
        f'<div class="card-title">{icon} {title}</div>'
        f'<div class="card-desc">{desc}</div></div>',
        target,
    )


def navcard(col, icon, title, desc, accent=None):
    accent = accent or CYAN["500"]
    _html(
        f'<div class="navcard" style="--accent:{accent};">'
        f'<div class="navcard-icon">{icon}</div>'
        f'<div class="navcard-title">{title}</div>'
        f'<div class="navcard-desc">{desc}</div></div>',
        col,
    )


def section(title, sub="", accent=None):
    accent = accent or CYAN["500"]
    _html(
        f'<div class="sec"><div class="sec-bar" style="--accent:{accent};"></div>'
        f'<div class="sec-title">{title}</div></div>'
    )
    if sub:
        _html(f'<div class="sec-sub">{sub}</div>')


TONES = {
    "info": (CYAN["600"], CYAN["50"]),
    "success": (d.COLOR["good"], "#eefaee"),
    "warning": (d.COLOR["warning"], "#fff8e8"),
    "danger": (d.COLOR["critical"], "#fdeeee"),
    "neutral": (d.COLOR["muted"], d.COLOR["page"]),
}


def info(text, tone="info"):
    accent, bg = TONES.get(tone, TONES["info"])
    st.markdown(
        f'<div class="ibox" style="--accent:{accent}; --bg:{bg};">{text}</div>',
        unsafe_allow_html=True,
    )


def badge(text, bg=None, fg=None, bd=None):
    bg = bg or CYAN["100"]
    fg = fg or CYAN["800"]
    bd = bd or CYAN["200"]
    return f'<span class="badge" style="--bg:{bg}; --fg:{fg}; --bd:{bd};">{text}</span>'


def progress_bar(value, accent=None):
    """value 0..1 -> HTML bar (dipakai di dalam card)."""
    accent = accent or CYAN["500"]
    pct = max(0.0, min(1.0, float(value))) * 100
    return (
        f'<div class="prog-wrap"><div class="prog-fill" '
        f'style="--accent:{accent}; width:{pct:.1f}%;"></div></div>'
    )
