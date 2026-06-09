# ============================================================
#  app.py — Descriptive Analytics Workbench  (Frontend Only)
#  All data logic lives in processing.py
#  Run: streamlit run app.py
# ============================================================

import io
import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Import all backend logic ──────────────────────────────
from processing import (
    load_file, merge_files, df_to_excel_bytes, df_to_pickle_bytes,
    detect_col_types, apply_renames, apply_dtype_changes, apply_filter,
    iqr_fences, numeric_full_stats,
    dtype_quality_table, missing_summary, duplicate_rows,
    descriptive_stats, percentile_table, skewness_kurtosis,
    normality_test, outlier_iqr_table, outlier_zscore_table, variance_table,
    frequency_table, rare_categories, shannon_entropy,
    correlation_pairs,
    remove_duplicates, impute_column, treat_outliers,
    apply_transforms, encoding_preview,
    build_recommendations,
    OUTLIER_DETECTION_INFO, OUTLIER_ACTION_INFO, ENCODING_INFO,
    get_best_encoding_recommendation,
)

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  PAGE CONFIG & STYLES
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Analytics Workbench",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg: #f8fafc;
    --surface: #ffffff;
    --surface2: #f1f5f9;
    --border: #e2e8f0;
    --accent: #6366f1;
    --accent2: #8b5cf6;
    --accent-light: #eef2ff;
    --success: #16a34a;
    --success-light: #f0fdf4;
    --warning: #d97706;
    --warning-light: #fffbeb;
    --danger: #dc2626;
    --danger-light: #fef2f2;
    --text: #0f172a;
    --text2: #334155;
    --muted: #64748b;
    --shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md: 0 4px 6px rgba(0,0,0,0.07), 0 2px 4px rgba(0,0,0,0.04);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp { background: var(--bg) !important; }
.block-container { padding: 1.5rem 2rem !important; }

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface2) !important;
    border-radius: 10px;
    padding: 4px;
    border: 1px solid var(--border);
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em;
    color: var(--muted) !important;
    border-radius: 7px !important;
    padding: 8px 14px !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: white !important;
}

/* ── Section Header ── */
.section-header {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--accent);
    border-left: 3px solid var(--accent);
    padding: 8px 14px;
    background: var(--accent-light);
    border-radius: 0 8px 8px 0;
    margin: 20px 0 14px 0;
}

/* ── Metric Card ── */
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
    box-shadow: var(--shadow);
}
.metric-card .val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent);
    display: block;
}
.metric-card .lbl {
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* ── Badges ── */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}
.badge-ok      { background: var(--success-light); color: var(--success);  border: 1px solid #bbf7d0; }
.badge-warn    { background: var(--warning-light);  color: var(--warning);  border: 1px solid #fde68a; }
.badge-danger  { background: var(--danger-light);   color: var(--danger);   border: 1px solid #fecaca; }
.badge-info    { background: var(--accent-light);   color: var(--accent);   border: 1px solid #c7d2fe; }

/* ── Info / Method Cards ── */
.method-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 14px;
    box-shadow: var(--shadow);
}
.method-card h4 {
    margin: 0 0 8px 0 !important;
    color: var(--accent);
    font-size: 0.9rem;
    font-weight: 700;
}
.method-card p, .method-card li {
    font-size: 0.85rem;
    color: var(--text2);
    margin: 4px 0;
}
.method-card .tag {
    display: inline-block;
    background: var(--surface2);
    color: var(--muted);
    border-radius: 6px;
    padding: 2px 8px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    margin: 2px 2px 0 0;
}

/* ── Rec Cards ── */
.rec-card        { background: linear-gradient(135deg,#f0f9ff,#e0f2fe); border:1px solid #bae6fd; border-radius:10px; padding:14px 16px; margin-bottom:10px; }
.rec-card-warn   { background: linear-gradient(135deg,#fffbeb,#fef3c7); border:1px solid #fde68a; border-radius:10px; padding:14px 16px; margin-bottom:10px; }
.rec-card-danger { background: linear-gradient(135deg,#fff1f2,#ffe4e6); border:1px solid #fecdd3; border-radius:10px; padding:14px 16px; margin-bottom:10px; }
.rec-card-ok     { background: linear-gradient(135deg,#f0fdf4,#dcfce7); border:1px solid #bbf7d0; border-radius:10px; padding:14px 16px; margin-bottom:10px; }

/* ── Progress Bar ── */
.progress-wrap { background: var(--surface2); border-radius: 8px; height: 8px; overflow: hidden; margin: 5px 0; }
.progress-fill { height: 100%; border-radius: 8px; }

/* ── Score Panel ── */
.score-breakdown-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 14px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    margin: 6px 0;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

code {
    font-family: 'JetBrains Mono', monospace;
    background: var(--surface2);
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.82rem;
    color: var(--accent);
}

div.stAlert { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None
if "clean_log" not in st.session_state:
    st.session_state.clean_log = []


# ─────────────────────────────────────────────
#  HELPER RENDERERS
# ─────────────────────────────────────────────

def render_rec_card(level, title, body):
    cls_map  = {"critical": "rec-card-danger", "warning": "rec-card-warn",
                "ok": "rec-card-ok", "info": "rec-card"}
    icon_map = {"critical": "🔴", "warning": "🟡", "ok": "🟢", "info": "🔵"}
    st.markdown(
        f'<div class="{cls_map.get(level, "rec-card")}">'
        f'<b>{icon_map.get(level, "ℹ️")} {title}</b><br>'
        f'<span style="font-size:0.85rem;color:#334155;">{body}</span></div>',
        unsafe_allow_html=True,
    )


def metric_card(val, label, color="var(--accent)"):
    st.markdown(
        f'<div class="metric-card"><span class="val" style="color:{color};">{val}</span>'
        f'<span class="lbl">{label}</span></div>',
        unsafe_allow_html=True,
    )


def render_method_card(name, info_dict):
    """Render a styled info card for an outlier or encoding method."""
    pros = "".join(f'<li>✅ {p}</li>' for p in info_dict.get("pros", []))
    cons = "".join(f'<li>❌ {c}</li>' for c in info_dict.get("cons", []))
    best = "".join(f'<span class="tag">{b}</span>' for b in info_dict.get("best_for", []))
    avoid = f'<p><b>⚠️ Avoid when:</b> {info_dict["avoid"]}</p>' if "avoid" in info_dict else ""

    st.markdown(f"""
    <div class="method-card">
        <h4>📌 {name}</h4>
        <p><b>What it does:</b> {info_dict.get('what', info_dict.get('full_name',''))}</p>
        <p><b>Use when:</b> {info_dict.get('when', '')}</p>
        <p><b>Why:</b> {info_dict.get('why', info_dict.get('effect', ''))}</p>
        {avoid}
        {"<ul>" + pros + cons + "</ul>" if pros or cons else ""}
        {('<p><b>Best for:</b> ' + best + '</p>') if best else ''}
    </div>
    """, unsafe_allow_html=True)


def quality_gauge(score):
    """Render a Plotly gauge for the quality score."""
    color = "#16a34a" if score >= 80 else "#d97706" if score >= 60 else "#dc2626"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Data Quality Score", "font": {"size": 16, "family": "Inter"}},
        number={"font": {"color": color, "size": 52, "family": "JetBrains Mono"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "#f1f5f9",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  40], "color": "#fee2e2"},
                {"range": [40, 70], "color": "#fef9c3"},
                {"range": [70, 100], "color": "#dcfce7"},
            ],
            "threshold": {
                "line": {"color": "#1e293b", "width": 3},
                "thickness": 0.75,
                "value": score,
            },
        },
    ))
    fig.update_layout(
        template="plotly_white",
        height=260,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="#ffffff",
    )
    return fig


def quality_breakdown_bars(score_dict, max_dict):
    """Render per-factor progress bars for the quality scorecard."""
    color_fn = lambda p: "#16a34a" if p >= 80 else "#d97706" if p >= 60 else "#dc2626"
    for factor, score_val in score_dict.items():
        max_val = max_dict.get(factor, 100)
        pct     = round(score_val / max_val * 100, 1)
        color   = color_fn(pct)
        st.markdown(f"""
        <div class="score-breakdown-row">
            <span style="font-size:0.85rem;color:#334155;font-weight:500;">{factor}</span>
            <div style="flex:1;margin:0 16px;">
                <div class="progress-wrap">
                    <div class="progress-fill" style="width:{pct}%;background:{color};"></div>
                </div>
            </div>
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;
                         color:{color};font-weight:700;min-width:60px;text-align:right;">
                {round(score_val,1)}&nbsp;/&nbsp;{max_dict[factor]}
            </span>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:20px 0 24px;">
        <div style="font-size:1.3rem;font-weight:700;color:#6366f1;">📊 Analytics Workbench</div>
        <div style="font-size:0.75rem;color:#64748b;margin-top:4px;">Descriptive EDA · Clean · Export</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## 📂 Dataset")
    upload_mode = st.radio("Mode", ["Single File", "Multiple Files (Merge)"], horizontal=True)

    df_raw = None

    if upload_mode == "Single File":
        uploaded = st.file_uploader("Upload CSV / Excel", type=["csv", "xlsx", "xls"])
        if uploaded:
            result = load_file(uploaded)
            if isinstance(result, tuple):
                xl, sheet_names = result
                sheet  = st.selectbox("Select Sheet", sheet_names)
                df_raw = xl.parse(sheet)
            else:
                df_raw = result
    else:
        files = st.file_uploader("Upload Multiple CSVs/Excels", type=["csv","xlsx","xls"],
                                  accept_multiple_files=True)
        if files:
            how = st.selectbox("Merge Strategy", ["Vertical (concat)", "Horizontal (concat)"])
            try:
                df_raw = merge_files(files, how="vertical" if "Vertical" in how else "horizontal")
                st.success(f"Merged {len(files)} files → {df_raw.shape}")
            except Exception as e:
                st.error(f"Merge failed: {e}")

    if df_raw is not None:
        if st.session_state.df_clean is None or st.session_state.df_clean.shape != df_raw.shape:
            st.session_state.df_clean = df_raw.copy()
            st.session_state.clean_log = []

        st.markdown("---")
        st.markdown("### ⚙️ Column Controls")
        all_cols = df_raw.columns.tolist()
        exclude  = st.multiselect("Exclude Columns", all_cols)
        keep     = [c for c in all_cols if c not in exclude]
        df_work  = st.session_state.df_clean[keep].copy()

        rename_mode = st.checkbox("Rename Columns")
        if rename_mode:
            renames = {}
            for c in keep:
                new_name = st.text_input(f"Rename '{c}'", value=c, key=f"ren_{c}")
                if new_name != c:
                    renames[c] = new_name
            if renames:
                df_work = apply_renames(df_work, renames)

        dtype_mode = st.checkbox("Change Data Types")
        if dtype_mode:
            dtype_map = {}
            for c in df_work.columns:
                cur    = str(df_work[c].dtype)
                choice = st.selectbox(
                    f"'{c}' ({cur})", ["(keep)","numeric","string","category","datetime"],
                    key=f"dt_{c}"
                )
                if choice != "(keep)":
                    dtype_map[c] = choice
            if dtype_map:
                df_work = apply_dtype_changes(df_work, dtype_map)

        st.markdown("---")
        st.markdown("### 🔍 Row Filters")
        filter_query = st.text_area(
            "Custom Filter (pandas query)",
            placeholder='e.g.  Age > 25 and Income < 100000',
            height=80,
        )
        if filter_query:
            try:
                df_work = apply_filter(df_work, filter_query)
                st.success(f"Filter applied — {len(df_work)} rows remain")
            except Exception as e:
                st.error(f"Filter error: {e}")

        num_cols, cat_cols, date_cols, bool_cols = detect_col_types(df_work)

        st.markdown("---")
        st.markdown("### 🎨 Chart Theme")
        palette   = st.selectbox("Color Palette", ["viridis","plasma","Set2","tab10","coolwarm","Blues","RdYlGn"])
        dark_mode = st.toggle("Dark Mode", value=False)
        chart_h   = st.slider("Chart Height (px)", 300, 900, 500, 50)
        chart_w   = st.slider("Chart Width (px)",  500, 1400, 900, 50)
        font_sz   = st.slider("Font Size", 8, 20, 12)
        show_grid = st.toggle("Show Grid", value=True)
        template  = "plotly_dark" if dark_mode else "plotly_white"

        # ── Live quality snapshot in sidebar ──────────────────
        st.markdown("---")
        rec_sidebar = build_recommendations(df_work, num_cols, cat_cols)
        sc_s = rec_sidebar["scorecard"]
        qc   = "#16a34a" if sc_s["score"] >= 80 else "#d97706" if sc_s["score"] >= 60 else "#dc2626"
        st.markdown(f"""
        <div style="background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:16px;">
            <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;">Live Dataset Stats</div>
            <div style="display:flex;justify-content:space-between;margin:5px 0;">
                <span style="font-size:0.82rem;color:#64748b;">Rows</span>
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.82rem;font-weight:700;">{len(df_work):,}</span>
            </div>
            <div style="display:flex;justify-content:space-between;margin:5px 0;">
                <span style="font-size:0.82rem;color:#64748b;">Columns</span>
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.82rem;font-weight:700;">{len(df_work.columns)}</span>
            </div>
            <div style="display:flex;justify-content:space-between;margin:5px 0;">
                <span style="font-size:0.82rem;color:#64748b;">Missing</span>
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.82rem;font-weight:700;color:#d97706;">{df_work.isnull().sum().sum():,}</span>
            </div>
            <div style="display:flex;justify-content:space-between;margin:5px 0;">
                <span style="font-size:0.82rem;color:#64748b;">Duplicates</span>
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.82rem;font-weight:700;color:#dc2626;">{rec_sidebar["duplicates"]["count"]}</span>
            </div>
            <div style="margin-top:12px;">
                <div style="font-size:0.7rem;color:#64748b;margin-bottom:5px;">Quality Score</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:1.5rem;font-weight:700;color:{qc};">{sc_s["score"]}<span style="font-size:0.85rem;font-weight:400;color:#94a3b8;">/100</span></div>
                <div style="background:#f1f5f9;border-radius:6px;height:6px;margin-top:5px;">
                    <div style="width:{sc_s['score']}%;height:100%;background:{qc};border-radius:6px;"></div>
                </div>
                <div style="font-size:0.72rem;color:{qc};font-weight:600;margin-top:4px;">Grade: {sc_s["grade"]} &nbsp;·&nbsp; {sc_s["critical"]} critical &nbsp;·&nbsp; {sc_s["warning"]} warnings</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MAIN AREA
# ─────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#4338ca 0%,#6366f1 60%,#8b5cf6 100%);
            border-radius:16px;padding:28px 32px;margin-bottom:24px;">
    <h1 style="font-family:'Inter',sans-serif;font-size:1.8rem;font-weight:700;color:#fff;margin:0;">
        📊 Descriptive Analytics Workbench
    </h1>
    <p style="color:rgba(255,255,255,0.75);margin:6px 0 0;font-size:0.95rem;">
        Upload → Explore → Clean → Transform → Export · Stay ML-ready.
    </p>
</div>
""", unsafe_allow_html=True)

if df_raw is None:
    st.info("⬅️  Upload a CSV or Excel file from the sidebar to get started.")
    st.stop()

num_cols, cat_cols, date_cols, bool_cols = detect_col_types(df_work)

tabs = st.tabs([
    "🗂 Overview",
    "🔢 Numerical",
    "🏷️ Categorical",
    "📐 Correlation",
    "📊 Charts",
    "🔄 Transform",
    "🧹 Data Cleaning",
    "💾 Export",
    "💡 Recommendations",
])


# ══════════════════════════════════════════════
#  TAB 1 — OVERVIEW
# ══════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-header">Dataset Snapshot</div>', unsafe_allow_html=True)

    cols_m = st.columns(5)
    for col_w, label, val in zip(cols_m,
        ["Rows", "Columns", "Numeric", "Categorical", "Missing %"],
        [f"{df_work.shape[0]:,}", f"{df_work.shape[1]:,}", str(len(num_cols)),
         str(len(cat_cols)),
         f"{df_work.isnull().sum().sum() / df_work.size * 100:.1f}%"]):
        with col_w:
            metric_card(val, label)

    st.markdown("&nbsp;")
    st.dataframe(df_work.head(50), use_container_width=True)

    st.markdown('<div class="section-header">Data Types & Quality</div>', unsafe_allow_html=True)
    st.dataframe(dtype_quality_table(df_work), use_container_width=True)

    st.markdown('<div class="section-header">Missing Values</div>', unsafe_allow_html=True)
    miss = missing_summary(df_work)
    if miss.empty:
        st.markdown('<span class="badge badge-ok">✅ No missing values found</span>', unsafe_allow_html=True)
    else:
        fig = px.bar(x=miss.index, y=miss.values,
                     labels={"x": "Column", "y": "Missing Count"},
                     color=miss.values, color_continuous_scale=palette,
                     template=template, height=int(chart_h // 1.5))
        fig.update_layout(font_size=font_sz, showlegend=False,
                          paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Duplicate Analysis</div>', unsafe_allow_html=True)
    dupes = duplicate_rows(df_work)
    c_d1, c_d2 = st.columns(2)
    with c_d1:
        metric_card(
            dupes,
            "Duplicate Rows",
            "#dc2626" if dupes > 0 else "#16a34a",
        )
    with c_d2:
        metric_card(
            f"{dupes/len(df_work)*100:.1f}%" if dupes else "0%",
            "Duplicate %",
            "#dc2626" if dupes > 0 else "#16a34a",
        )
    if dupes:
        st.dataframe(df_work[df_work.duplicated()], use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 2 — NUMERICAL
# ══════════════════════════════════════════════
with tabs[1]:
    if not num_cols:
        st.warning("No numeric columns detected.")
    else:
        sel_num = st.multiselect("Select Numeric Columns", num_cols,
                                  default=num_cols[:min(8, len(num_cols))])
        analyses = st.multiselect("Select Analyses", [
            "Descriptive Statistics", "Percentile Table", "Skewness & Kurtosis",
            "Normality Test (Shapiro-Wilk)", "Outlier Detection (IQR)",
            "Outlier Detection (Z-Score)", "Variance Analysis",
        ], default=["Descriptive Statistics"])

        if sel_num:
            if "Descriptive Statistics" in analyses:
                st.markdown('<div class="section-header">Descriptive Statistics</div>', unsafe_allow_html=True)
                st.dataframe(descriptive_stats(df_work, sel_num).style.format("{:.4f}"),
                             use_container_width=True)

            if "Percentile Table" in analyses:
                st.markdown('<div class="section-header">Percentile Table</div>', unsafe_allow_html=True)
                st.dataframe(percentile_table(df_work, sel_num).style.format("{:.4f}"),
                             use_container_width=True)

            if "Skewness & Kurtosis" in analyses:
                st.markdown('<div class="section-header">Skewness & Kurtosis</div>', unsafe_allow_html=True)
                st.dataframe(skewness_kurtosis(df_work, sel_num), use_container_width=True)

            if "Normality Test (Shapiro-Wilk)" in analyses:
                st.markdown('<div class="section-header">Normality Test</div>', unsafe_allow_html=True)
                st.dataframe(normality_test(df_work, sel_num), use_container_width=True)

            if "Outlier Detection (IQR)" in analyses:
                st.markdown('<div class="section-header">Outlier Detection — IQR</div>', unsafe_allow_html=True)
                st.dataframe(outlier_iqr_table(df_work, sel_num), use_container_width=True)

            if "Outlier Detection (Z-Score)" in analyses:
                st.markdown('<div class="section-header">Outlier Detection — Z-Score</div>', unsafe_allow_html=True)
                st.dataframe(outlier_zscore_table(df_work, sel_num), use_container_width=True)

            if "Variance Analysis" in analyses:
                st.markdown('<div class="section-header">Variance Analysis</div>', unsafe_allow_html=True)
                st.dataframe(variance_table(df_work, sel_num), use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 3 — CATEGORICAL
# ══════════════════════════════════════════════
with tabs[2]:
    if not cat_cols:
        st.warning("No categorical columns detected.")
    else:
        sel_cat = st.multiselect("Select Categorical Columns", cat_cols,
                                  default=cat_cols[:min(5, len(cat_cols))])
        cat_analyses = st.multiselect("Select Analyses", [
            "Frequency Table", "Cardinality", "Dominant Category",
            "Rare Categories (<1%)", "Entropy Analysis",
        ], default=["Frequency Table", "Cardinality"])

        for col in sel_cat:
            st.markdown(f'<div class="section-header">📌 {col}</div>', unsafe_allow_html=True)
            s  = df_work[col].dropna()
            vc = s.value_counts()

            if "Frequency Table" in cat_analyses:
                st.dataframe(frequency_table(s), use_container_width=True)

            if "Cardinality" in cat_analyses:
                n_u = s.nunique()
                badge_cls = "badge-warn" if n_u > 50 else "badge-ok"
                st.markdown(
                    f'Unique values: <span class="badge {badge_cls}">{n_u} &nbsp; '
                    f'{"High Cardinality ⚠️" if n_u > 50 else "Normal ✅"}</span>',
                    unsafe_allow_html=True,
                )

            if "Dominant Category" in cat_analyses:
                vcp = s.value_counts(normalize=True) * 100
                dom = vc.idxmax()
                st.info(f"**Dominant:** `{dom}` → {vc.max():,} rows ({vcp.max():.1f}%)")

            if "Rare Categories (<1%)" in cat_analyses:
                rare = rare_categories(s)
                if rare.empty:
                    st.markdown('<span class="badge badge-ok">✅ No rare categories</span>', unsafe_allow_html=True)
                else:
                    st.warning(f"{len(rare)} rare categories (<1%):")
                    st.dataframe(rare)

            if "Entropy Analysis" in cat_analyses:
                ent, max_ent = shannon_entropy(s)
                st.metric(f"Shannon Entropy ('{col}')", f"{ent:.4f} bits",
                          delta=f"Max possible: {max_ent:.2f} bits")


# ══════════════════════════════════════════════
#  TAB 4 — CORRELATION
# ══════════════════════════════════════════════
with tabs[3]:
    if len(num_cols) < 2:
        st.warning("Need at least 2 numeric columns for correlation.")
    else:
        corr_cols   = st.multiselect("Select Columns", num_cols, default=num_cols)
        corr_method = st.selectbox("Method", ["pearson", "spearman", "kendall"])
        sig_level   = st.slider("Significance Level α", 0.01, 0.10, 0.05, 0.01)

        if len(corr_cols) >= 2:
            corr_df = df_work[corr_cols].corr(method=corr_method)

            st.markdown('<div class="section-header">Correlation Matrix</div>', unsafe_allow_html=True)
            st.dataframe(corr_df.style.background_gradient(cmap=palette, axis=None).format("{:.3f}"),
                         use_container_width=True)

            fig_c = px.imshow(corr_df, text_auto=".2f", color_continuous_scale=palette,
                              template=template, height=chart_h, width=chart_w,
                              title=f"{corr_method.capitalize()} Correlation Heatmap")
            fig_c.update_layout(font_size=font_sz)
            st.plotly_chart(fig_c, use_container_width=True)

            st.markdown('<div class="section-header">Top Correlations (Ranked)</div>', unsafe_allow_html=True)
            st.dataframe(
                correlation_pairs(df_work, corr_cols, method=corr_method, sig_level=sig_level),
                use_container_width=True,
            )


# ══════════════════════════════════════════════
#  TAB 5 — CHARTS
# ══════════════════════════════════════════════
with tabs[4]:
    chart_type = st.selectbox("Chart Type", [
        "Histogram", "KDE / Density", "Box Plot", "Violin Plot",
        "Scatter Plot", "Bar Chart", "Pie / Donut Chart",
        "Line Chart", "Area Chart", "Pair Plot",
        "ECDF Plot", "QQ Plot", "Heatmap (Correlation)",
    ])
    col_x          = st.selectbox("X / Primary Column", df_work.columns.tolist())
    col_y          = st.selectbox("Y Column (if needed)", ["None"] + df_work.columns.tolist())
    col_color      = st.selectbox("Color By (optional)", ["None"] + cat_cols)
    chart_title    = st.text_input("Chart Title", value=chart_type)
    x_label        = st.text_input("X-Axis Label", value=col_x)
    y_label        = st.text_input("Y-Axis Label", value=col_y if col_y != "None" else "")
    legend_visible = st.toggle("Show Legend", value=True)

    color_arg = None if col_color == "None" else col_color
    y_arg     = None if col_y    == "None" else col_y
    fig_chart = None

    try:
        if chart_type == "Histogram":
            fig_chart = px.histogram(df_work, x=col_x, color=color_arg,
                template=template, height=chart_h, title=chart_title, nbins=40)

        elif chart_type == "KDE / Density":
            from scipy.stats import gaussian_kde
            vals = df_work[col_x].dropna().values
            kde  = gaussian_kde(vals)
            xs   = np.linspace(vals.min(), vals.max(), 300)
            fig_chart = px.line(x=xs, y=kde(xs), labels={"x": x_label, "y": "Density"},
                template=template, height=chart_h, title=chart_title)

        elif chart_type == "Box Plot":
            fig_chart = px.box(df_work, x=color_arg, y=col_x, color=color_arg,
                template=template, height=chart_h, title=chart_title)

        elif chart_type == "Violin Plot":
            fig_chart = px.violin(df_work, x=color_arg, y=col_x, color=color_arg,
                box=True, points="outliers", template=template, height=chart_h, title=chart_title)

        elif chart_type == "Scatter Plot" and y_arg:
            fig_chart = px.scatter(df_work, x=col_x, y=y_arg, color=color_arg,
                template=template, height=chart_h, title=chart_title, trendline="ols")

        elif chart_type == "Bar Chart":
            gb = df_work[col_x].value_counts().reset_index()
            gb.columns = [col_x, "count"]
            fig_chart = px.bar(gb, x=col_x, y="count", color=col_x,
                template=template, height=chart_h, title=chart_title)

        elif chart_type == "Pie / Donut Chart":
            vc = df_work[col_x].value_counts().reset_index()
            vc.columns = ["label", "count"]
            fig_chart = px.pie(vc, names="label", values="count", hole=0.4,
                template=template, height=chart_h, title=chart_title)

        elif chart_type == "Line Chart" and y_arg:
            fig_chart = px.line(df_work, x=col_x, y=y_arg, color=color_arg,
                template=template, height=chart_h, title=chart_title)

        elif chart_type == "Area Chart" and y_arg:
            fig_chart = px.area(df_work, x=col_x, y=y_arg, color=color_arg,
                template=template, height=chart_h, title=chart_title)

        elif chart_type == "ECDF Plot":
            sorted_vals = np.sort(df_work[col_x].dropna())
            ecdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            fig_chart = px.line(x=sorted_vals, y=ecdf, labels={"x": x_label, "y": "ECDF"},
                template=template, height=chart_h, title=chart_title)

        elif chart_type == "QQ Plot":
            vals = df_work[col_x].dropna().values
            (osm, osr), (slope, intercept, _r) = stats.probplot(vals, dist="norm")
            fig_chart = go.Figure()
            fig_chart.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name="Data"))
            fig_chart.add_trace(go.Scatter(
                x=osm, y=slope * np.array(osm) + intercept,
                mode="lines", name="Normal Line", line=dict(color="red", dash="dash")
            ))
            fig_chart.update_layout(
                template=template, height=chart_h, title=chart_title,
                xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles",
            )

        elif chart_type == "Pair Plot":
            sel_pp = st.multiselect("Choose Columns for Pair Plot", num_cols,
                                     default=num_cols[:min(5, len(num_cols))])
            if sel_pp:
                fig_chart = px.scatter_matrix(df_work, dimensions=sel_pp, color=color_arg,
                    template=template, height=chart_h, title=chart_title)

        elif chart_type == "Heatmap (Correlation)":
            c_df = df_work[num_cols].corr()
            fig_chart = px.imshow(c_df, text_auto=".2f", color_continuous_scale=palette,
                template=template, height=chart_h, title=chart_title)

        if fig_chart:
            fig_chart.update_layout(
                font_size=font_sz, showlegend=legend_visible,
                xaxis_title=x_label, yaxis_title=y_label,
                xaxis=dict(showgrid=show_grid), yaxis=dict(showgrid=show_grid),
            )
            st.plotly_chart(fig_chart, use_container_width=True)

            fmt = st.selectbox("Export Chart As", ["png", "svg", "html"])
            if st.button("⬇️ Download Chart"):
                if fmt == "html":
                    buf = io.StringIO()
                    fig_chart.write_html(buf)
                    st.download_button("Download HTML", buf.getvalue(),
                                       file_name=f"{chart_title}.html")
                else:
                    img_bytes = fig_chart.to_image(format=fmt)
                    st.download_button(f"Download {fmt.upper()}", img_bytes,
                        file_name=f"{chart_title}.{fmt}", mime=f"image/{fmt}")
        else:
            st.info("Select valid columns for this chart type.")

    except Exception as e:
        st.error(f"Chart error: {e}")


# ══════════════════════════════════════════════
#  TAB 6 — TRANSFORM
# ══════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-header">Column Transformations</div>', unsafe_allow_html=True)
    if not num_cols:
        st.warning("No numeric columns to transform.")
    else:
        t_col = st.selectbox("Select Column to Transform", num_cols)
        transforms = st.multiselect("Transformations to Apply", [
            "Log (log1p)", "Square Root", "Box-Cox", "Yeo-Johnson",
            "Standard Scaling (Z-score)", "MinMax Scaling [0,1]", "Robust Scaling",
        ])

        if t_col and transforms:
            df_transformed, preview, errors = apply_transforms(df_work, t_col, transforms)
            if preview:
                st.dataframe(pd.DataFrame(preview).T, use_container_width=True)
                st.success("Transformations applied. Download from the Export tab.")
            for t_name, err_msg in errors.items():
                st.warning(f"Could not apply '{t_name}': {err_msg}")


# ══════════════════════════════════════════════
#  TAB 7 — DATA CLEANING
# ══════════════════════════════════════════════
with tabs[6]:
    st.markdown("### 🧹 Data Cleaning Operations")
    st.caption("All operations modify the working dataset and are logged below.")

    df_clean_work = st.session_state.df_clean[keep].copy()
    num_c, cat_c, _, _ = detect_col_types(df_clean_work)

    # ── A: Duplicate Removal ──────────────────────────────────
    st.markdown('<div class="section-header">🔁 Duplicate Row Removal</div>', unsafe_allow_html=True)
    dup_count = duplicate_rows(df_clean_work)
    col_a, col_b = st.columns([2, 1])
    with col_a:
        dup_subset = st.multiselect(
            "Check duplicates based on columns (leave empty = all columns)",
            df_clean_work.columns.tolist(), key="dup_subset",
        )
        dup_keep = st.selectbox("Which duplicate to keep?",
            ["first", "last", "none (drop all)"], key="dup_keep")
    with col_b:
        badge_cls = "badge-danger" if dup_count > 0 else "badge-ok"
        st.markdown(f"""
        <div style="margin-top:10px;">
            <div style="font-size:0.75rem;color:#64748b;margin-bottom:6px;">Duplicate Rows</div>
            <span class="badge {badge_cls}">{dup_count} {'found ⚠️' if dup_count else 'none ✅'}</span>
        </div>
        """, unsafe_allow_html=True)

    if st.button("🗑️ Remove Duplicates", disabled=(dup_count == 0), key="btn_dup"):
        subset_arg = dup_subset if dup_subset else None
        keep_arg   = False if "none" in dup_keep else dup_keep
        df_new, removed = remove_duplicates(st.session_state.df_clean,
                                            subset=subset_arg, keep=keep_arg)
        st.session_state.df_clean = df_new
        st.session_state.clean_log.append(
            f"✅ Removed {removed} duplicate rows (keep='{dup_keep}')")
        st.success(f"Removed {removed} rows. Dataset now has {len(df_new):,} rows.")
        st.rerun()

    st.divider()

    # ── B: Missing Value Imputation ───────────────────────────
    st.markdown('<div class="section-header">🕳️ Missing Value Imputation</div>', unsafe_allow_html=True)
    miss_df = df_clean_work.isna().sum()
    miss_df = miss_df[miss_df > 0]

    if miss_df.empty:
        st.markdown('<span class="badge badge-ok">✅ No missing values in the dataset!</span>',
                    unsafe_allow_html=True)
    else:
        st.info(f"**{miss_df.sum():,} missing values** across **{len(miss_df)} columns**")
        st.dataframe(pd.DataFrame({
            "Column":        miss_df.index,
            "Missing Count": miss_df.values,
            "Missing %":     (miss_df / len(df_clean_work) * 100).round(2).values,
        }), use_container_width=True)

        st.markdown("**Select imputation strategy per column:**")
        impute_actions = {}
        for col in miss_df.index:
            is_num = col in num_c
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"`{col}` — **{'numeric' if is_num else 'categorical'}** — "
                            f"{miss_df[col]} missing ({miss_df[col]/len(df_clean_work)*100:.1f}%)")
            with col2:
                if is_num:
                    method = st.selectbox(
                        f"Strategy for '{col}'",
                        ["(skip)", "Fill with Mean", "Fill with Median", "Fill with Mode",
                         "Fill with Constant", "Fill with Forward Fill (ffill)",
                         "Fill with Backward Fill (bfill)", "Drop rows with missing"],
                        key=f"miss_{col}",
                    )
                    const_val = None
                    if method == "Fill with Constant":
                        const_val = st.number_input(f"Constant for '{col}'",
                                                    key=f"const_{col}", value=0.0)
                else:
                    method = st.selectbox(
                        f"Strategy for '{col}'",
                        ["(skip)", "Fill with Mode", "Fill with Constant",
                         "Fill with Forward Fill (ffill)", "Fill with Backward Fill (bfill)",
                         "Drop rows with missing"],
                        key=f"miss_{col}",
                    )
                    const_val = None
                    if method == "Fill with Constant":
                        const_val = st.text_input(f"Constant for '{col}'",
                                                  key=f"const_{col}", value="Unknown")
            impute_actions[col] = (method, const_val)

        _METHOD_MAP = {
            "Fill with Mean":                   "mean",
            "Fill with Median":                 "median",
            "Fill with Mode":                   "mode",
            "Fill with Constant":               "constant",
            "Fill with Forward Fill (ffill)":   "ffill",
            "Fill with Backward Fill (bfill)":  "bfill",
            "Drop rows with missing":           "drop",
        }

        if st.button("✅ Apply Imputation", key="btn_impute"):
            df_imp = st.session_state.df_clean.copy()
            applied = []
            for col, (method, const_val) in impute_actions.items():
                if method == "(skip)" or col not in df_imp.columns:
                    continue
                backend_method = _METHOD_MAP.get(method)
                if backend_method:
                    df_imp = impute_column(df_imp, col, backend_method, constant=const_val)
                    applied.append(f"{col} → {method}")
            st.session_state.df_clean = df_imp
            for a in applied:
                st.session_state.clean_log.append(f"✅ Imputed: {a}")
            st.success(f"Applied imputation to {len(applied)} column(s).")
            st.rerun()

    st.divider()

    # ── C: Outlier Treatment ──────────────────────────────────
    st.markdown('<div class="section-header">📌 Outlier Detection & Treatment</div>',
                unsafe_allow_html=True)

    # ── Method info expanders ──────────────────────────────────
    with st.expander("📖 Learn about outlier detection methods", expanded=False):
        tabs_out = st.tabs(list(OUTLIER_DETECTION_INFO.keys()))
        for tab_o, (method_name, info) in zip(tabs_out, OUTLIER_DETECTION_INFO.items()):
            with tab_o:
                render_method_card(method_name, info)

    with st.expander("📖 Learn about outlier treatment actions", expanded=False):
        tabs_act = st.tabs(list(OUTLIER_ACTION_INFO.keys()))
        for tab_a, (action_name, info) in zip(tabs_act, OUTLIER_ACTION_INFO.items()):
            with tab_a:
                render_method_card(action_name, info)

    if not num_c:
        st.info("No numeric columns available for outlier treatment.")
    else:
        out_col = st.selectbox("Select Numeric Column", num_c, key="out_col")
        s_out   = df_clean_work[out_col].dropna()
        lo, hi, _ = iqr_fences(s_out)
        n_iqr = int(((s_out < lo) | (s_out > hi)).sum())
        n_z   = int((np.abs(stats.zscore(s_out)) > 3).sum())

        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card(n_iqr, "IQR Outliers",  "#dc2626" if n_iqr > 0 else "#16a34a")
        with c2:
            metric_card(n_z,   "Z-Score Outliers", "#dc2626" if n_z > 0 else "#16a34a")
        with c3:
            metric_card(f"{lo:.2f} / {hi:.2f}", "Lower / Upper Fence", "#6366f1")

        out_method = st.selectbox("Detection Method",
            list(OUTLIER_DETECTION_INFO.keys()), key="out_method")
        out_action = st.selectbox("Treatment Action",
            list(OUTLIER_ACTION_INFO.keys()), key="out_action")

        _ACTION_MAP = {
            "Cap / Winsorise (clip to fences)": "cap",
            "Fill with Mean":                   "mean",
            "Fill with Median":                 "median",
            "Fill with Mode":                   "mode",
            "Remove rows with outliers":        "remove",
        }
        _DET_MAP = {
            "IQR (1.5×IQR)":      "iqr",
            "Z-Score (|z| > 3)":  "zscore",
        }

        if st.button(f"⚡ Apply Outlier Treatment to '{out_col}'", key="btn_out"):
            df_new, n_affected, msg = treat_outliers(
                st.session_state.df_clean, out_col,
                detection=_DET_MAP[out_method],
                action=_ACTION_MAP[out_action],
            )
            st.session_state.df_clean = df_new
            st.session_state.clean_log.append(f"✅ {msg}")
            st.success(msg)
            st.rerun()

    st.divider()

    # ── D: Cleaning Log ───────────────────────────────────────
    st.markdown('<div class="section-header">📋 Cleaning Log</div>', unsafe_allow_html=True)
    if not st.session_state.clean_log:
        st.info("No cleaning operations performed yet.")
    else:
        for i, entry in enumerate(st.session_state.clean_log, 1):
            st.markdown(f"**{i}.** {entry}")
        if st.button("🔄 Reset to Original Data", key="btn_reset"):
            st.session_state.df_clean = df_raw.copy()
            st.session_state.clean_log = []
            st.success("Dataset reset to original.")
            st.rerun()

    st.divider()

    # ── E: ML-Ready / Encoding Preview ────────────────────────
    st.markdown('<div class="section-header">🤖 Encoding Preview & ML-Ready Export</div>',
                unsafe_allow_html=True)

    # ── Encoding method info expanders ────────────────────────
    with st.expander("📖 Learn about encoding methods — which one to choose?", expanded=False):
        enc_tabs = st.tabs(list(ENCODING_INFO.keys()))
        for tab_e, (enc_name, info) in zip(enc_tabs, ENCODING_INFO.items()):
            with tab_e:
                render_method_card(enc_name, info)

    if cat_cols:
        enc_col = st.selectbox("Column to Preview", cat_cols, key="enc_col_select")

        if enc_col in cat_cols:
            # Smart recommendation
            enc_rec = get_best_encoding_recommendation(df_work[enc_col])
            rec_color = "#16a34a" if enc_rec["n_unique"] <= 15 else "#d97706"
            st.markdown(f"""
            <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;
                        padding:14px 16px;margin:10px 0;">
                <b style="color:#16a34a;">🤖 Recommended: {enc_rec['recommended']}</b><br>
                <span style="font-size:0.85rem;color:#334155;">{enc_rec['reason']}</span><br>
                <span style="font-size:0.78rem;color:#64748b;">
                    Unique values: <b style="color:{rec_color};">{enc_rec['n_unique']}</b>
                    &nbsp;·&nbsp; {enc_rec['pct_unique']}% of rows are unique
                </span>
            </div>
            """, unsafe_allow_html=True)

            enc_method = st.selectbox(
                "Encoding Method (preview only)",
                ["Label Encoding", "One-Hot Encoding", "Ordinal Encoding", "Frequency Encoding"],
                index=["Label Encoding","One-Hot Encoding","Ordinal Encoding","Frequency Encoding"].index(
                    enc_rec["recommended"]) if enc_rec["recommended"] in
                    ["Label Encoding","One-Hot Encoding","Ordinal Encoding","Frequency Encoding"] else 0,
                key="enc_method_select",
            )
            preview_enc = encoding_preview(df_work[enc_col], enc_col, enc_method)
            st.dataframe(preview_enc.head(30), use_container_width=True)
    else:
        st.info("No categorical columns found for encoding preview.")

    if st.button("💾 Save ML-Ready as .pkl"):
        miss_pct_ml = df_work.isna().mean() * 100
        payload = {
            "dataframe": df_work, "num_cols": num_cols,
            "cat_cols":  cat_cols, "date_cols": date_cols, "shape": df_work.shape,
            "missing_pct": miss_pct_ml.to_dict(),
        }
        pkl_buf = df_to_pickle_bytes(payload)
        st.download_button("⬇️ Download ml_ready.pkl", pkl_buf,
            file_name="ml_ready.pkl", mime="application/octet-stream")


# ══════════════════════════════════════════════
#  TAB 8 — EXPORT
# ══════════════════════════════════════════════
with tabs[7]:
    st.markdown('<div class="section-header">Export Data</div>', unsafe_allow_html=True)

    export_choice = st.radio("Export Which Dataset?",
        ["Original (filtered/renamed)", "Cleaned (after Data Cleaning tab)"])
    exp_df = df_work if export_choice.startswith("Original") else st.session_state.df_clean[keep]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("⬇️ Download CSV", exp_df.to_csv(index=False),
            file_name="analytics_export.csv", mime="text/csv")
    with c2:
        st.download_button("⬇️ Download Excel", df_to_excel_bytes(exp_df),
            file_name="analytics_export.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with c3:
        st.download_button("⬇️ Download JSON", exp_df.to_json(orient="records"),
            file_name="analytics_export.json", mime="application/json")

    st.markdown('<div class="section-header">Export Summary Statistics</div>', unsafe_allow_html=True)
    if num_cols:
        summary = exp_df[num_cols].describe().T
        st.download_button("⬇️ Summary Stats (CSV)", summary.to_csv(),
            file_name="summary_stats.csv", mime="text/csv")


# ══════════════════════════════════════════════
#  TAB 9 — RECOMMENDATIONS  (with quality gauge)
# ══════════════════════════════════════════════
with tabs[8]:
    st.markdown("### 💡 Automated Data Recommendations")
    st.caption("Scans your dataset and gives step-by-step, actionable guidance with exact Python code.")

    rec = build_recommendations(df_work, num_cols, cat_cols)
    all_stats  = rec["_all_stats"]
    miss_pct   = rec["_miss_pct"]
    sc         = rec["scorecard"]

    # ── Quality Gauge + Scorecard ─────────────────────────────
    st.markdown('<div class="section-header">📊 Data Quality Scorecard</div>', unsafe_allow_html=True)

    g_col, b_col = st.columns([1, 1])

    with g_col:
        st.plotly_chart(quality_gauge(sc["score"]), use_container_width=True)

    with b_col:
        st.markdown("&nbsp;")
        qc_top = st.columns(4)
        for qw, label, val, col_hex in zip(
            qc_top,
            ["Score", "Grade", "Critical", "Warnings"],
            [f"{sc['score']}/100", sc["grade"], sc["critical"], sc["warning"]],
            ["#6366f1", "#16a34a" if sc["grade"] in ("A","B") else "#d97706",
             "#dc2626" if sc["critical"] else "#16a34a",
             "#d97706" if sc["warning"] else "#16a34a"],
        ):
            with qw:
                metric_card(val, label, col_hex)

        st.markdown("&nbsp;")

        # Compute per-factor scores for the breakdown bars
        miss_pct_v  = df_work.isnull().sum().sum() / df_work.size * 100
        dup_pct_v   = df_work.duplicated().sum() / len(df_work) * 100
        out_info_v  = rec.get("outliers", [])
        avg_out_pct = np.mean([o["pct"] for o in out_info_v]) if out_info_v else 0
        inv_cnt     = sum(r["n_iqr"] for r in out_info_v) if out_info_v else 0
        inv_pct_v   = inv_cnt / len(df_work) * 100 if len(df_work) else 0

        factor_scores = {
            "Missing Values (30 pts)":  max(0, 30 - miss_pct_v * 0.6),
            "Duplicates (20 pts)":      max(0, 20 - dup_pct_v  * 0.4),
            "Outliers (20 pts)":        max(0, 20 - avg_out_pct * 0.4),
            "Invalid / Domain (15 pts)": max(0, 15 - inv_pct_v * 0.3),
            "Consistency (15 pts)":     15.0,
        }
        factor_max = {
            "Missing Values (30 pts)": 30,
            "Duplicates (20 pts)":     20,
            "Outliers (20 pts)":       20,
            "Invalid / Domain (15 pts)": 15,
            "Consistency (15 pts)":    15,
        }
        quality_breakdown_bars(factor_scores, factor_max)

    # ── Overall verdict ───────────────────────────────────────
    total_issues = sc["critical"] + sc["warning"]
    if sc["score"] >= 90:
        st.success("🎉 Excellent dataset quality — ready for analysis or modelling!")
    elif sc["score"] >= 75:
        st.info(f"📈 Good quality. Address {total_issues} issue(s) before modelling.")
    elif sc["score"] >= 60:
        st.warning(f"⚠️ Moderate quality. {sc['critical']} critical issues need attention.")
    else:
        st.error(f"🚨 Poor data quality. {sc['critical']} critical issues must be fixed before modelling.")

    st.markdown("---")

    # 1. Duplicates ───────────────────────────────────────────
    st.markdown('<div class="section-header">1️⃣ Duplicate Rows</div>', unsafe_allow_html=True)
    d = rec["duplicates"]
    if d["count"] == 0:
        render_rec_card("ok", "No duplicate rows", "Dataset is free of duplicate rows. ✅")
    else:
        render_rec_card(
            "critical" if d["pct"] > 5 else "warning",
            f"{d['count']:,} duplicate rows detected ({d['pct']}%)",
            f"Duplicates skew statistics and inflate model performance.<br>"
            f"<b>Fix:</b> Go to 🧹 Data Cleaning → <i>Duplicate Removal</i>, or run:<br>"
            f"<code>df.drop_duplicates(inplace=True)</code>")

    # 2. Missing Values ────────────────────────────────────────
    st.markdown('<div class="section-header">2️⃣ Missing Values</div>', unsafe_allow_html=True)
    if not rec["missing"]:
        render_rec_card("ok", "No missing values", "All columns are complete. ✅")
    for m in rec["missing"]:
        col   = m["col"]
        pct_m = m["pct"]
        if pct_m > 30:
            render_rec_card("critical", f"'{col}' has {pct_m:.1f}% missing — consider dropping",
                f"More than 30% missing — this column may not be reliable.<br>"
                f"<b>Options:</b><br>"
                f"• Drop column: <code>df.drop('{col}', axis=1, inplace=True)</code><br>"
                f"• Impute: <code>df['{col}'].fillna(df['{col}'].median(), inplace=True)</code>")
        elif m["is_numeric"]:
            skew_v   = m["skewness"]
            rec_fill = "Median" if skew_v and abs(skew_v) > 1 else "Mean"
            code_val = f"df['{col}'].median()" if rec_fill == "Median" else f"df['{col}'].mean()"
            render_rec_card("warning", f"'{col}' has {pct_m:.1f}% missing",
                f"Skewness = {skew_v} → recommend <b>{rec_fill}</b> imputation.<br>"
                f"<code>df['{col}'].fillna({code_val}, inplace=True)</code>")
        else:
            render_rec_card("warning", f"'{col}' has {pct_m:.1f}% missing (categorical)",
                f"<b>Options:</b><br>"
                f"• Mode: <code>df['{col}'].fillna(df['{col}'].mode()[0], inplace=True)</code><br>"
                f"• Unknown: <code>df['{col}'].fillna('Unknown', inplace=True)</code>")

    # 3. Skewness ─────────────────────────────────────────────
    st.markdown('<div class="section-header">3️⃣ Skewness & Normality</div>', unsafe_allow_html=True)
    if not rec["skewness"]:
        render_rec_card("ok", "All numeric columns have acceptable skewness",
                        "No highly skewed columns detected. ✅")
    for s_r in rec["skewness"]:
        col    = s_r["col"]
        skew_v = s_r["skewness"]
        direction = "right (+)" if skew_v > 0 else "left (–)"
        render_rec_card("warning", f"'{col}' is highly skewed (skewness = {skew_v})",
            f"Skewed {direction}. Affects regression and mean-based statistics.<br>"
            f"<b>Fix:</b><br>"
            f"1. Log: <code>df['{col}_log'] = np.log1p(df['{col}'])</code><br>"
            f"2. Yeo-Johnson: <code>from sklearn.preprocessing import PowerTransformer<br>"
            f"   pt = PowerTransformer(method='yeo-johnson')<br>"
            f"   df['{col}_yj'] = pt.fit_transform(df[['{col}']])</code><br>"
            f"3. Re-check: <code>df['{col}_log'].skew()</code> — aim for |skew| &lt; 0.5")

    # 4. Outliers ─────────────────────────────────────────────
    st.markdown('<div class="section-header">4️⃣ Outlier Detection</div>', unsafe_allow_html=True)
    if not rec["outliers"]:
        render_rec_card("ok", "No significant outliers detected (IQR method)",
                        "All numeric columns look clean. ✅")
    for o in rec["outliers"]:
        col  = o["col"]
        lo_f = o["lo"]; hi_f = o["hi"]
        render_rec_card(
            "critical" if o["pct"] >= 5 else "warning",
            f"'{col}' has {o['n_iqr']} IQR outlier(s) ({o['pct']}%)",
            f"Fences: [{lo_f:.3f}, {hi_f:.3f}]  |  Z-score outliers: {o['n_z']}<br>"
            f"<b>Options:</b><br>"
            f"• Cap: <code>df['{col}'] = df['{col}'].clip(lower={lo_f:.3f}, upper={hi_f:.3f})</code><br>"
            f"• Fill median: <code>mask=(df['{col}']&lt;{lo_f:.3f})|(df['{col}']&gt;{hi_f:.3f})<br>"
            f"  df.loc[mask,'{col}']=df['{col}'].median()</code><br>"
            f"• Remove: <code>df=df[~mask].reset_index(drop=True)</code>")

    # 5. Cardinality ──────────────────────────────────────────
    st.markdown('<div class="section-header">5️⃣ Categorical Cardinality</div>', unsafe_allow_html=True)
    card_high  = [r for r in rec["cardinality"] if r["n_unique"] > 50]
    card_const = [r for r in rec["cardinality"] if r["n_unique"] == 1]
    if not card_high and not card_const:
        render_rec_card("ok", "Categorical cardinality is acceptable",
                        "All categorical columns have manageable unique counts. ✅")
    for r in card_const:
        col = r["col"]
        render_rec_card("critical", f"'{col}' has only 1 unique value — constant column",
            f"Drop it: <code>df.drop('{col}', axis=1, inplace=True)</code>")
    for r in card_high:
        col = r["col"]; n = r["n_unique"]
        render_rec_card("warning", f"'{col}' has high cardinality ({n} unique values)",
            f"<b>Better approaches:</b><br>"
            f"• Frequency encode: <code>freq=df['{col}'].value_counts(normalize=True)<br>"
            f"  df['{col}_freq']=df['{col}'].map(freq)</code><br>"
            f"• Group rare: <code>top=df['{col}'].value_counts().head(20).index<br>"
            f"  df['{col}']=df['{col}'].where(df['{col}'].isin(top),other='Other')</code>")

    # 6. Low Variance ─────────────────────────────────────────
    st.markdown('<div class="section-header">6️⃣ Low / Zero Variance Columns</div>', unsafe_allow_html=True)
    if not rec["low_variance"]:
        render_rec_card("ok", "All numeric columns have adequate variance", "✅")
    for r in rec["low_variance"]:
        col = r["col"]
        if r["std"] < 1e-6:
            render_rec_card("critical", f"'{col}' has zero variance (constant)",
                f"Drop it: <code>df.drop('{col}', axis=1, inplace=True)</code>")
        else:
            render_rec_card("info", f"'{col}' has very low variance (CV = {r['cv_pct']}%)",
                "Consider whether this column is meaningful before including in models.")

    # 7. Class Imbalance ──────────────────────────────────────
    st.markdown('<div class="section-header">7️⃣ Class Distribution (Categorical)</div>', unsafe_allow_html=True)
    if not rec["imbalance"]:
        render_rec_card("ok", "No severe class imbalance detected",
                        "Categorical columns look balanced. ✅")
    for r in rec["imbalance"]:
        col = r["col"]; dom_val = r["dominant_val"]; pct_i = r["dominant_pct"]
        render_rec_card(
            "critical" if pct_i >= 90 else "warning",
            f"'{col}' dominated by '{dom_val}' ({pct_i:.1f}%)",
            f"If this is a target/label column, the dataset is class-imbalanced.<br>"
            f"<b>Fixes:</b><br>"
            f"• Oversample: <code>from imblearn.over_sampling import SMOTE</code><br>"
            f"• Use <code>class_weight='balanced'</code> in sklearn models<br>"
            f"• Evaluate with F1, AUC-ROC instead of accuracy")

    # 8. Multicollinearity ────────────────────────────────────
    if len(num_cols) >= 2:
        st.markdown('<div class="section-header">8️⃣ Multicollinearity (High Correlation)</div>',
                    unsafe_allow_html=True)
        if not rec["high_corr"]:
            render_rec_card("ok", "No highly correlated pairs (r > 0.85)",
                            "No multicollinearity risk detected. ✅")
        for r in rec["high_corr"]:
            c1n = r["col_a"]; c2n = r["col_b"]; r_val = r["r"]
            render_rec_card("warning", f"High correlation: '{c1n}' ↔ '{c2n}' (r = {r_val})",
                f"Multicollinearity can destabilise linear models.<br>"
                f"<b>Options:</b><br>"
                f"• Drop one: <code>df.drop('{c2n}', axis=1, inplace=True)</code><br>"
                f"• Use PCA to combine correlated features<br>"
                f"• Use regularisation (Ridge/Lasso)")

    # 9. EDA Checklist ────────────────────────────────────────
    st.markdown('<div class="section-header">✅ Complete EDA Checklist</div>', unsafe_allow_html=True)
    for item, done in rec["checklist"]:
        badge_cls = "badge-ok" if done else "badge-warn"
        icon      = "✅" if done else "⬜"
        st.markdown(
            f'{icon} {item} &nbsp; <span class="badge {badge_cls}">{"Done" if done else "Pending"}</span>',
            unsafe_allow_html=True,
        )
