# ============================================================
#  app.py — Streamlit UI Layer
#  All backend logic lives in preprocessing.py
#  Run: streamlit run app.py
# ============================================================

import io
import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import matplotlib; matplotlib.use("Agg")

# ── Import ALL backend logic from preprocessing.py ──────────
from preprocessing import (
    # DB
    init_db, log_op, get_history,
    # File
    load_file, check_merge_schema,
    # Column types
    col_types,
    # Stats
    iqr_bounds, full_stats, descriptive_stats, percentile_table,
    normality_test, outlier_summary_iqr, outlier_summary_zscore,
    skewness_table, variance_table, correlation_table, categorical_summary,
    # Validation
    validate_value, validate_row,
    # Cleaning
    remove_duplicates, drop_rows_query, keep_rows_query,
    drop_rows_by_value, drop_rows_by_index, change_dtype,
    # Nulls
    fill_nulls, fill_nulls_bulk, drop_null_rows,
    # Outliers
    get_outlier_mask, treat_outliers,
    # Transforms
    apply_transform,
    # Similarity
    detect_similar_cols,
    # Quality
    quality_score, quality_breakdown,
    # Recommendations
    generate_recommendations, recommendations_score,
    # Export
    to_excel_bytes, to_csv_bytes, to_json_bytes, add_new_row,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DataPrep Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif!important;background:#f8f9fc!important;color:#111827!important}
.stApp{background:#f8f9fc!important}
[data-testid="stSidebar"]{background:#ffffff!important;border-right:1px solid #e2e6f0!important}
[data-testid="stSidebar"] *{color:#111827!important}
.main-header{background:linear-gradient(135deg,#1e3a8a,#2563eb 60%,#3b82f6);border-radius:14px;padding:24px 28px;margin-bottom:20px}
.main-header h1{font-size:1.6rem!important;font-weight:700!important;color:#fff!important;margin:0!important}
.main-header p{color:rgba(255,255,255,.75)!important;margin:4px 0 0!important;font-size:.9rem}
.metric-card{background:#fff;border:1px solid #e2e6f0;border-radius:12px;padding:18px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.07)}
.metric-card .val{font-family:'JetBrains Mono',monospace;font-size:1.6rem;font-weight:700;color:#2563eb;display:block}
.metric-card .label{font-size:.72rem;color:#6b7280;text-transform:uppercase;letter-spacing:1px;margin-top:3px}
.section-header{display:flex;align-items:center;padding:10px 16px;background:#eff6ff;border-left:3px solid #2563eb;border-radius:0 8px 8px 0;margin:18px 0 12px}
.section-header h3{margin:0!important;font-size:.9rem;font-weight:600;color:#2563eb}
.badge{display:inline-block;padding:2px 10px;border-radius:20px;font-size:.73rem;font-weight:600;font-family:'JetBrains Mono',monospace}
.badge-ok{background:#f0fdf4;color:#16a34a;border:1px solid #bbf7d0}
.badge-warn{background:#fffbeb;color:#d97706;border:1px solid #fde68a}
.badge-danger{background:#fef2f2;color:#dc2626;border:1px solid #fecaca}
.badge-info{background:#eff6ff;color:#2563eb;border:1px solid #bfdbfe}
.rec-info{background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;padding:14px;margin:6px 0}
.rec-warn{background:#fffbeb;border:1px solid #fde68a;border-radius:10px;padding:14px;margin:6px 0}
.rec-danger{background:#fef2f2;border:1px solid #fecaca;border-radius:10px;padding:14px;margin:6px 0}
.rec-ok{background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;padding:14px;margin:6px 0}
.stButton>button{background:linear-gradient(135deg,#2563eb,#3b82f6)!important;color:#fff!important;border:none!important;border-radius:8px!important;font-weight:600!important;padding:9px 22px!important}
.stButton>button:hover{opacity:.9!important}
.stTabs [data-baseweb="tab-list"]{background:#f1f3f9!important;border-radius:10px;padding:3px;border:1px solid #e2e6f0}
.stTabs [data-baseweb="tab"]{color:#6b7280!important;border-radius:6px!important;font-weight:500;font-size:.82rem}
.stTabs [aria-selected="true"]{background:#2563eb!important;color:#fff!important}
code{font-family:'JetBrains Mono',monospace;font-size:.8rem;background:#f1f5f9;padding:2px 5px;border-radius:4px}
.sidebar-stat{display:flex;justify-content:space-between;margin:5px 0;font-size:.84rem}
.sidebar-stat .sv{font-family:'JetBrains Mono',monospace;font-weight:700}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  INIT
# ─────────────────────────────────────────────
init_db()

for k, v in {"df": None, "original_df": None, "fname": "", "log": []}.items():
    if k not in st.session_state:
        st.session_state[k] = v


def slog(msg: str):
    """Append to session log AND SQLite history."""
    from datetime import datetime
    st.session_state.log.append(
        f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    )
    log_op(st.session_state.fname, msg)


# ─────────────────────────────────────────────
#  SIDEBAR — UPLOAD
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0 20px'>
      <div style='font-size:1.25rem;font-weight:700;color:#2563eb'>📊 DataPrep Pro</div>
      <div style='font-size:.72rem;color:#6b7280;margin-top:3px'>Smart Preprocessing Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    mode = st.radio("Upload Mode", ["Single File", "Merge Multiple Files"], horizontal=True)
    df_raw = None

    if mode == "Single File":
        uf = st.file_uploader("CSV or Excel", type=["csv", "xlsx", "xls"])
        if uf:
            raw = uf.read()
            name = uf.name
            if name.lower().endswith((".xlsx", ".xls")):
                xl = pd.ExcelFile(io.BytesIO(raw))
                sheet = st.selectbox("Sheet", xl.sheet_names)
                df_raw = xl.parse(sheet)
            else:
                df_raw = load_file(io.BytesIO(raw), name)
            st.session_state.fname = name
    else:
        files = st.file_uploader(
            "Multiple CSV/Excel", type=["csv", "xlsx", "xls"], accept_multiple_files=True
        )
        if files:
            frames, names = [], []
            for ff in files:
                raw = ff.read()
                try:
                    fr = load_file(io.BytesIO(raw), ff.name)
                    frames.append(fr)
                    names.append(ff.name)
                except Exception as e:
                    st.error(f"{ff.name}: {e}")
            if frames:
                mismatches = check_merge_schema(frames, names)
                if mismatches:
                    st.warning(
                        f"⚠️ {mismatches} have different columns — "
                        f"merge may create NaN values."
                    )
                how = st.selectbox("Merge", ["Vertical (stack rows)", "Horizontal (join cols)"])
                try:
                    df_raw = pd.concat(
                        frames,
                        axis=0 if "Vertical" in how else 1,
                        ignore_index=True
                    )
                    st.success(f"Merged {len(frames)} files → {df_raw.shape}")
                    st.session_state.fname = "merged_file.csv"
                except Exception as e:
                    st.error(str(e))

    if df_raw is not None:
        if st.session_state.df is None or st.button("🔄 Load New File"):
            st.session_state.df = df_raw.copy()
            st.session_state.original_df = df_raw.copy()
            st.session_state.log = []
            slog(f"Loaded: {st.session_state.fname} — {df_raw.shape}")

    st.markdown("---")

    if st.session_state.df is not None:
        df_sb = st.session_state.df
        qs = quality_score(df_sb)
        qc = "#16a34a" if qs >= 80 else "#d97706" if qs >= 60 else "#dc2626"
        n_miss = df_sb.isnull().sum().sum()
        n_dup  = df_sb.duplicated().sum()
        st.markdown(f"""
        <div style='background:#fff;border:1px solid #e2e6f0;border-radius:12px;padding:14px'>
          <div style='font-size:.68rem;color:#9ca3af;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px'>Live Stats</div>
          <div class='sidebar-stat'><span style='color:#6b7280'>Rows</span><span class='sv'>{len(df_sb):,}</span></div>
          <div class='sidebar-stat'><span style='color:#6b7280'>Columns</span><span class='sv'>{len(df_sb.columns)}</span></div>
          <div class='sidebar-stat'><span style='color:#6b7280'>Missing</span>
            <span class='sv' style='color:#d97706'>{n_miss:,}</span></div>
          <div class='sidebar-stat'><span style='color:#6b7280'>Duplicates</span>
            <span class='sv' style='color:{"#dc2626" if n_dup>0 else "#16a34a"}'>{n_dup}</span></div>
          <div style='margin-top:12px;font-size:.68rem;color:#9ca3af'>Quality Score</div>
          <div style='font-family:"JetBrains Mono",monospace;font-size:1.5rem;font-weight:700;color:{qc}'>{qs}<span style='font-size:.8rem;color:#9ca3af'>/100</span></div>
          <div style='background:#f1f3f9;border-radius:6px;height:6px;margin-top:5px'>
            <div style='width:{qs}%;height:100%;background:{qc};border-radius:6px'></div></div>
        </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Chart Settings**")
        palette   = st.selectbox("Palette", ["viridis","plasma","Set2","tab10","coolwarm","Blues"], label_visibility="collapsed")
        dark_mode = st.toggle("Dark Mode", False)
        chart_h   = st.slider("Height", 300, 900, 460, 50)
        font_sz   = st.slider("Font", 8, 20, 12)
        show_grid = st.toggle("Grid", True)
        template  = "plotly_dark" if dark_mode else "plotly_white"
    else:
        palette = "viridis"; dark_mode = False; chart_h = 460
        font_sz = 12;        show_grid = True;  template = "plotly_white"

# ─────────────────────────────────────────────
#  MAIN HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
  <h1>📊 DataPrep Pro — Smart Analytics Dashboard</h1>
  <p>Upload → Inspect → Clean → Analyse → Export. Descriptive analysis, no ML model needed.</p>
</div>
""", unsafe_allow_html=True)

if st.session_state.df is None:
    st.info("⬅️  Upload a CSV or Excel file from the sidebar to get started.")
    st.stop()

# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
TABS = st.tabs([
    "🗂 Overview",
    "🧹 Cleaning",
    "🚫 Null Handler",
    "📦 Outlier Handler",
    "➕ Add Row",
    "🔢 Numerical",
    "🏷️ Categorical",
    "📐 Correlation",
    "📊 Charts",
    "🔄 Transform",
    "💡 Recommendations",
    "💾 Export & History",
])

# ══════════════════════════════════════════════
#  TAB 0 — OVERVIEW
# ══════════════════════════════════════════════
with TABS[0]:
    df = st.session_state.df
    num_cols, cat_cols, date_cols, bool_cols = col_types(df)

    st.markdown("<div class='section-header'><h3>Dataset Snapshot</h3></div>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    for w, lab, val in zip([c1,c2,c3,c4,c5],
        ["Rows","Columns","Numeric","Categorical","Missing"],
        [f"{df.shape[0]:,}", str(df.shape[1]), len(num_cols),
         len(cat_cols), f"{df.isnull().sum().sum():,}"]):
        with w:
            st.markdown(
                f"<div class='metric-card'><span class='val'>{val}</span>"
                f"<span class='label'>{lab}</span></div>", unsafe_allow_html=True
            )

    st.markdown("&nbsp;")
    n_preview = st.select_slider("Rows to preview", [5, 10, 20, 50, 100, "All"], value=20)
    show = df if n_preview == "All" else df.head(int(n_preview))
    st.dataframe(show, use_container_width=True, height=340)

    st.markdown("<div class='section-header'><h3>Column Info</h3></div>", unsafe_allow_html=True)
    mem = df.memory_usage(deep=True)
    info_df = pd.DataFrame({
        "Column":      df.columns,
        "Dtype":       df.dtypes.astype(str).values,
        "Non-Null":    df.notna().sum().values,
        "Null Count":  df.isnull().sum().values,
        "Null %":      (df.isnull().mean() * 100).round(2).values,
        "Unique":      df.nunique().values,
        "Memory KB":   [round(mem.get(c, 0) / 1024, 2) for c in df.columns],
        "Sample":      [str(df[c].dropna().iloc[0])[:30]
                        if df[c].notna().any() else "—" for c in df.columns],
    })
    st.dataframe(info_df, use_container_width=True)

    miss = df.isnull().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if miss.empty:
        st.markdown("<span class='badge badge-ok'>✅ No missing values</span>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='section-header'><h3>Missing Values Chart</h3></div>", unsafe_allow_html=True)
        fig_m = px.bar(
            x=miss.index, y=(miss / len(df) * 100).round(2),
            labels={"x": "Column", "y": "Missing %"},
            color=(miss / len(df) * 100).values,
            color_continuous_scale=palette, template=template, height=300,
        )
        fig_m.update_layout(font_size=font_sz, showlegend=False)
        st.plotly_chart(fig_m, use_container_width=True)

# ══════════════════════════════════════════════
#  TAB 1 — CLEANING
# ══════════════════════════════════════════════
with TABS[1]:
    df = st.session_state.df
    num_cols, cat_cols, date_cols, bool_cols = col_types(df)

    # ── Duplicates ────────────────────────────
    st.markdown("<div class='section-header'><h3>🔁 Duplicate Rows</h3></div>", unsafe_allow_html=True)
    n_dup = df.duplicated().sum()
    c1d, c2d = st.columns([3, 1])
    with c1d:
        dup_sub  = st.multiselect("Check by columns (empty = all)", df.columns.tolist(), key="dup_sub")
        dup_keep = st.radio("Keep", ["first", "last", "none (drop all)"], horizontal=True, key="dup_keep")
    with c2d:
        st.metric("Duplicates", n_dup, delta="✅ Clean" if n_dup == 0 else f"⚠️ {n_dup} found")

    if n_dup > 0:
        with st.expander(f"View {n_dup} duplicate rows"):
            st.dataframe(df[df.duplicated(keep="first")], use_container_width=True)

    if st.button("🗑️ Remove Duplicates", key="btn_dup", disabled=(n_dup == 0)):
        sub_arg  = dup_sub if dup_sub else None
        keep_arg = False if "none" in dup_keep else dup_keep
        new_df, removed = remove_duplicates(st.session_state.df, subset=sub_arg, keep=keep_arg)
        st.session_state.df = new_df
        slog(f"Removed {removed} duplicate rows")
        st.success(f"✅ Removed {removed} duplicates. Dataset: {len(new_df):,} rows.")
        st.rerun()

    # ── Drop rows ─────────────────────────────
    st.markdown("<div class='section-header'><h3>🗑️ Drop Rows</h3></div>", unsafe_allow_html=True)
    c1r, c2r = st.columns(2)
    with c1r:
        drop_q = st.text_area("Drop where (condition)", placeholder="e.g.  Age < 0  or  Salary > 999999",
                               height=70, key="drop_q")
        if st.button("🗑️ Drop Matching Rows") and drop_q:
            try:
                new_df, n_dropped = drop_rows_query(st.session_state.df, drop_q)
                st.session_state.df = new_df
                slog(f"Dropped {n_dropped} rows: {drop_q}")
                st.success(f"Dropped {n_dropped} rows."); st.rerun()
            except Exception as e:
                st.error(str(e))
    with c2r:
        keep_q = st.text_area("Keep where (condition)", placeholder="e.g.  Age >= 18 and Salary > 0",
                               height=70, key="keep_q")
        if st.button("✅ Apply Keep Filter") and keep_q:
            try:
                new_df, kept, removed = keep_rows_query(st.session_state.df, keep_q)
                st.session_state.df = new_df
                slog(f"Kept {kept} rows, removed {removed}: {keep_q}")
                st.success(f"Kept {kept:,} rows (removed {removed})."); st.rerun()
            except Exception as e:
                st.error(str(e))

    dc1, dc2 = st.columns(2)
    with dc1:
        drop_col_sel = st.selectbox("Drop by column value — column", df.columns.tolist(), key="dcs")
        drop_vals = st.multiselect(
            "Drop rows where value is",
            df[drop_col_sel].dropna().unique().tolist()[:100], key="dvs"
        )
        if st.button("🗑️ Drop by Value") and drop_vals:
            new_df, n_dropped = drop_rows_by_value(st.session_state.df, drop_col_sel, drop_vals)
            st.session_state.df = new_df
            slog(f"Dropped {n_dropped} rows where {drop_col_sel} in {drop_vals}")
            st.success(f"Dropped {n_dropped} rows."); st.rerun()
    with dc2:
        idx_inp = st.text_input("Drop by row index (comma-sep)", placeholder="0, 5, 12", key="idx_inp")
        if st.button("🗑️ Drop by Index") and idx_inp:
            try:
                idxs = [int(x.strip()) for x in idx_inp.split(",")]
                new_df, n_dropped = drop_rows_by_index(st.session_state.df, idxs)
                st.session_state.df = new_df
                slog(f"Dropped indices {idxs}")
                st.success(f"Dropped {len(idxs)} rows."); st.rerun()
            except Exception as e:
                st.error(str(e))

    # ── Column Controls ───────────────────────
    st.markdown("<div class='section-header'><h3>⚙️ Column Controls</h3></div>", unsafe_allow_html=True)
    cc1, cc2 = st.columns(2)
    with cc1:
        excl = st.multiselect("Drop Columns", df.columns.tolist(), key="excl_cols")
        if st.button("Drop Selected Columns") and excl:
            st.session_state.df.drop(columns=excl, inplace=True, errors="ignore")
            slog(f"Dropped columns: {excl}"); st.success(f"Dropped {excl}"); st.rerun()
    with cc2:
        rename_on = st.checkbox("Rename Columns")
        if rename_on:
            rmap = {}
            for c in df.columns:
                nv = st.text_input(f"'{c}'", value=c, key=f"ren_{c}")
                if nv != c:
                    rmap[c] = nv
            if rmap and st.button("Apply Renames"):
                st.session_state.df.rename(columns=rmap, inplace=True)
                slog(f"Renamed: {rmap}"); st.success("Renamed."); st.rerun()

    # ── Data Types ────────────────────────────
    st.markdown("<div class='section-header'><h3>🔄 Change Data Types</h3></div>", unsafe_allow_html=True)
    dtype_cols = st.multiselect("Select columns to retype", df.columns.tolist(), key="dtype_cols")
    if dtype_cols:
        for c in dtype_cols:
            cur = str(df[c].dtype)
            nd  = st.selectbox(
                f"'{c}' ({cur})",
                ["(keep)", "numeric", "string", "category", "datetime"],
                key=f"dt_{c}"
            )
            if nd != "(keep)" and st.button(f"Apply dtype to '{c}'", key=f"apply_dt_{c}"):
                try:
                    st.session_state.df = change_dtype(st.session_state.df, c, nd)
                    slog(f"Changed dtype '{c}' → {nd}"); st.success(f"'{c}' → {nd}"); st.rerun()
                except Exception as e:
                    st.error(str(e))

    # ── Similar Columns ───────────────────────
    st.markdown("<div class='section-header'><h3>🔍 Similar / Redundant Columns</h3></div>", unsafe_allow_html=True)
    with st.spinner("Checking column similarity…"):
        suggestions = detect_similar_cols(df)
    if suggestions:
        st.warning(f"Found {len(suggestions)} potentially redundant column pair(s).")
        for s in suggestions:
            with st.expander(f"**{s['col1']}** ↔ **{s['col2']}** — Score: {s['score']}%"):
                st.markdown(f"**Reason:** {s['reason']}")
                try:
                    samp = df[[s["col1"], s["col2"]]].dropna().head(8).reset_index(drop=True)
                    st.dataframe(samp, use_container_width=True)
                except Exception:
                    pass
                if st.button(f"Drop '{s['col2']}'", key=f"drop_sim_{s['col1']}_{s['col2']}"):
                    if s["col2"] in st.session_state.df.columns:
                        st.session_state.df.drop(columns=[s["col2"]], inplace=True)
                        slog(f"Dropped similar column: {s['col2']}"); st.rerun()
    else:
        st.markdown("<span class='badge badge-ok'>✅ No redundant columns detected</span>",
                    unsafe_allow_html=True)

    # ── Cleaning Log ──────────────────────────
    st.markdown("<div class='section-header'><h3>📋 Cleaning Log</h3></div>", unsafe_allow_html=True)
    if st.session_state.log:
        for e in st.session_state.log[-15:]:
            st.markdown(f"<span class='badge badge-info'>{e}</span>&nbsp;", unsafe_allow_html=True)
        if st.button("🔄 Reset to Original"):
            st.session_state.df = st.session_state.original_df.copy()
            st.session_state.log = []
            st.success("Reset to original."); st.rerun()
    else:
        st.info("No operations yet.")

# ══════════════════════════════════════════════
#  TAB 2 — NULL HANDLER
# ══════════════════════════════════════════════
with TABS[2]:
    df = st.session_state.df
    num_cols, cat_cols, _, _ = col_types(df)

    st.markdown("<div class='section-header'><h3>Missing Value Summary</h3></div>", unsafe_allow_html=True)
    miss_s = df.isnull().sum(); miss_s = miss_s[miss_s > 0].sort_values(ascending=False)
    if miss_s.empty:
        st.markdown("<span class='badge badge-ok'>✅ No missing values!</span>", unsafe_allow_html=True)
    else:
        mdf = pd.DataFrame({"Column": miss_s.index, "Missing": miss_s.values,
            "Missing %": (miss_s / len(df) * 100).round(2).values,
            "Dtype": [str(df[c].dtype) for c in miss_s.index]})
        st.dataframe(mdf, use_container_width=True)

    st.markdown("<div class='section-header'><h3>Fill / Impute</h3></div>", unsafe_allow_html=True)
    scope = st.radio("Apply to", ["One Column","All Numeric","All Categorical","All Columns"], horizontal=True)

    if scope == "One Column":
        imp_col  = st.selectbox("Column", df.columns.tolist(), key="imp_col")
        is_num_c = imp_col in num_cols
        methods_num = ["mean","median","mode","constant","ffill","bfill","interpolate","drop"]
        methods_cat = ["mode","constant","ffill","bfill","drop"]
        method_label_map = {
            "mean":"Mean","median":"Median","mode":"Mode","constant":"Constant Value",
            "ffill":"Forward Fill","bfill":"Backward Fill",
            "interpolate":"Interpolate (linear)","drop":"Drop rows"
        }
        choices = methods_num if is_num_c else methods_cat
        method = st.selectbox(
            "Method",
            choices,
            format_func=lambda x: method_label_map.get(x, x),
            key="imp_meth"
        )
        const = st.text_input("Constant value", "0", key="imp_const") if method == "constant" else None

        if st.button("✅ Fill", key="fill_one"):
            new_df, n_filled = fill_nulls(st.session_state.df, imp_col, method,
                                          const=(float(const) if (const and is_num_c) else const))
            st.session_state.df = new_df
            slog(f"Filled {n_filled} nulls in '{imp_col}' via {method}")
            st.success(f"Filled {n_filled} values in '{imp_col}'."); st.rerun()
    else:
        bulk_cols = (num_cols if scope == "All Numeric" else
                     cat_cols if scope == "All Categorical" else
                     df.columns.tolist())
        bulk_label_map = {
            "mean":"Mean (numeric only)","median":"Median (numeric only)","mode":"Mode",
            "constant":"Constant","ffill":"Forward Fill","bfill":"Backward Fill"
        }
        bulk_m = st.selectbox(
            "Bulk Method", list(bulk_label_map.keys()),
            format_func=lambda x: bulk_label_map[x], key="bulk_m"
        )
        bulk_c = st.text_input("Constant value", "0", key="bulk_const") if bulk_m == "constant" else None

        if st.button("✅ Fill All", key="fill_bulk"):
            new_df, total = fill_nulls_bulk(st.session_state.df, bulk_cols, bulk_m, const=bulk_c)
            st.session_state.df = new_df
            slog(f"Bulk fill ({bulk_m}) → {total} values filled across {len(bulk_cols)} cols")
            st.success(f"Filled {total} values across {len(bulk_cols)} columns."); st.rerun()

    st.markdown("<div class='section-header'><h3>Drop Rows with Missing Values</h3></div>",
                unsafe_allow_html=True)
    drop_null_cols  = st.multiselect("In columns (empty = any)", df.columns.tolist(), key="dnc")
    null_thresh_pct = st.slider("Or: drop rows missing more than % values", 0, 100, 50)
    cn1, cn2 = st.columns(2)
    with cn1:
        if st.button("🗑️ Drop Null Rows"):
            sub = drop_null_cols if drop_null_cols else None
            new_df, n_dropped = drop_null_rows(st.session_state.df, subset=sub)
            st.session_state.df = new_df
            slog(f"Dropped {n_dropped} null rows"); st.success(f"Dropped {n_dropped} rows."); st.rerun()
    with cn2:
        if st.button("🗑️ Drop by Null %"):
            new_df, n_dropped = drop_null_rows(st.session_state.df, threshold_pct=null_thresh_pct)
            st.session_state.df = new_df
            slog(f"Dropped {n_dropped} rows (>{null_thresh_pct}% null)")
            st.success(f"Dropped {n_dropped} rows."); st.rerun()

# ══════════════════════════════════════════════
#  TAB 3 — OUTLIER HANDLER
# ══════════════════════════════════════════════
with TABS[3]:
    df = st.session_state.df
    num_cols, *_ = col_types(df)

    if not num_cols:
        st.warning("No numeric columns found.")
    else:
        st.markdown("<div class='section-header'><h3>Outlier Summary (All Numeric Columns)</h3></div>",
                    unsafe_allow_html=True)
        out_method = st.radio("Detection Method", ["IQR", "Z-Score"], horizontal=True, key="out_meth")

        if out_method == "IQR":
            st.dataframe(outlier_summary_iqr(df, num_cols), use_container_width=True)
        else:
            st.dataframe(outlier_summary_zscore(df, num_cols), use_container_width=True)

        # Strip plot — all columns
        try:
            n_num = len(num_cols)
            fig_strip = make_subplots(rows=1, cols=n_num,
                                      subplot_titles=num_cols, horizontal_spacing=0.05)
            for i, c in enumerate(num_cols, 1):
                s = df[c].dropna()
                lo, hi, _ = iqr_bounds(s)
                mask_out = (s < lo) | (s > hi)
                fig_strip.add_trace(go.Strip(
                    y=s[~mask_out], name="Normal",
                    marker=dict(color="rgba(37,99,235,.5)", size=4),
                    showlegend=(i == 1)), row=1, col=i)
                if mask_out.sum():
                    fig_strip.add_trace(go.Strip(
                        y=s[mask_out], name="Outlier",
                        marker=dict(color="rgba(220,38,38,.8)", size=6),
                        showlegend=(i == 1)), row=1, col=i)
            fig_strip.update_layout(
                title="Outlier Strip Plot (Red = Outlier)",
                template=template, height=380,
                paper_bgcolor="#fff", plot_bgcolor="#f8f9fc"
            )
            st.plotly_chart(fig_strip, use_container_width=True)
        except Exception as e:
            st.error(f"Strip plot error: {e}")

        st.markdown("<div class='section-header'><h3>Treat Outliers</h3></div>", unsafe_allow_html=True)
        out_col_sel = st.selectbox("Column", num_cols, key="out_col_treat")
        out_action  = st.selectbox("Action", [
            "cap", "mean", "median", "mode", "nan", "remove"
        ], format_func=lambda x: {
            "cap": "Cap / Winsorise (clip to fences)",
            "mean": "Fill with Mean", "median": "Fill with Median",
            "mode": "Fill with Mode", "nan": "Replace with NaN (mark missing)",
            "remove": "Remove outlier rows"
        }[x], key="out_act")

        if out_col_sel:
            s = df[out_col_sel].dropna()
            lo, hi, _ = iqr_bounds(s)
            n_iqr = ((s < lo) | (s > hi)).sum()
            st.metric("IQR Outliers", n_iqr)
            st.caption(f"Bounds: [{lo:.4f}, {hi:.4f}]")
            fig_bx = px.box(df, y=out_col_sel, template=template,
                            height=300, title=f"Box Plot — {out_col_sel}")
            fig_bx.add_hline(y=lo, line_dash="dash", line_color="red",   annotation_text="Lower fence")
            fig_bx.add_hline(y=hi, line_dash="dash", line_color="green", annotation_text="Upper fence")
            fig_bx.update_layout(font_size=font_sz)
            st.plotly_chart(fig_bx, use_container_width=True)

        if st.button(f"⚡ Apply to '{out_col_sel}'", key="btn_out"):
            new_df, n_aff, msg = treat_outliers(
                st.session_state.df, out_col_sel,
                action=out_action,
                method=out_method.lower().replace("-", "")
            )
            st.session_state.df = new_df
            slog(msg); st.success(msg); st.rerun()

# ══════════════════════════════════════════════
#  TAB 4 — ADD ROW (User Input + Validation)
# ══════════════════════════════════════════════
with TABS[4]:
    df = st.session_state.df
    num_cols, cat_cols, _, _ = col_types(df)

    st.markdown("<div class='section-header'><h3>➕ Add New Row Manually</h3></div>",
                unsafe_allow_html=True)
    st.caption("Fields are validated against dtype, domain rules, and known categories. Invalid fields shown in red.")

    input_data = {}
    form_cols  = st.columns(min(3, len(df.columns)))
    for i, c in enumerate(df.columns):
        with form_cols[i % 3]:
            dtype = str(df[c].dtype)
            if "int" in dtype or "float" in dtype:
                input_data[c] = st.number_input(c, value=0.0, key=f"inp_{c}")
            elif c in cat_cols and df[c].nunique() < 50:
                opts = list(df[c].dropna().unique())
                input_data[c] = (st.selectbox(c, opts, key=f"inp_{c}")
                                 if opts else st.text_input(c, key=f"inp_{c}"))
            elif "bool" in dtype:
                input_data[c] = st.selectbox(c, [True, False], key=f"inp_{c}")
            else:
                input_data[c] = st.text_input(c, key=f"inp_{c}")

    if st.button("🔍 Validate & Add Row", key="btn_add_row"):
        val_results = validate_row(input_data, df)
        n_inv = sum(1 for r in val_results if r["Status"] == "Invalid")
        vdf   = pd.DataFrame(val_results)

        def _vrow(row):
            if row["Status"] == "Invalid":
                return ["background-color:#fef2f2;color:#dc2626"] * len(row)
            return ["background-color:#f0fdf4;color:#16a34a"] * len(row)

        try:
            st.dataframe(vdf.style.apply(_vrow, axis=1), use_container_width=True, hide_index=True)
        except Exception:
            st.dataframe(vdf, use_container_width=True)

        cv1, cv2 = st.columns(2)
        cv1.markdown(f"<span class='badge badge-ok'>✅ {len(val_results)-n_inv} Valid</span>",
                     unsafe_allow_html=True)
        cv2.markdown(f"<span class='badge badge-danger'>❌ {n_inv} Invalid</span>",
                     unsafe_allow_html=True)

        if n_inv == 0:
            st.session_state.df = add_new_row(st.session_state.df, input_data)
            slog(f"Added new row ({len(st.session_state.df)} rows total)")
            st.success(f"✅ Row added! Dataset now has {len(st.session_state.df):,} rows.")
            st.dataframe(st.session_state.df.tail(5), use_container_width=True)
        else:
            st.warning(f"⚠️ {n_inv} field(s) failed validation.")
            c_keep, c_info = st.columns(2)
            with c_keep:
                if st.button("✅ Add Anyway (ignore validation)", key="add_anyway"):
                    st.session_state.df = add_new_row(st.session_state.df, input_data)
                    slog(f"Added row with {n_inv} invalid field(s)")
                    st.success("Row added with warnings."); st.rerun()
            with c_info:
                st.info("Fix values above and click Validate & Add again.")

# ══════════════════════════════════════════════
#  TAB 5 — NUMERICAL ANALYSIS
# ══════════════════════════════════════════════
with TABS[5]:
    df = st.session_state.df
    num_cols, *_ = col_types(df)

    if not num_cols:
        st.warning("No numeric columns detected.")
    else:
        sel = st.multiselect("Columns", num_cols, default=num_cols[:min(8, len(num_cols))])
        analyses = st.multiselect("Analyses", [
            "Descriptive Statistics", "Percentile Table", "Skewness & Kurtosis",
            "Normality Test", "Outlier Summary (IQR)", "Outlier Summary (Z-Score)",
            "Variance Analysis"
        ], default=["Descriptive Statistics"])

        if sel and analyses:
            if "Descriptive Statistics" in analyses:
                st.markdown("<div class='section-header'><h3>Descriptive Statistics</h3></div>",
                            unsafe_allow_html=True)
                st.dataframe(descriptive_stats(df, sel).style.format("{:.4f}"),
                             use_container_width=True)

            if "Percentile Table" in analyses:
                st.markdown("<div class='section-header'><h3>Percentile Table</h3></div>",
                            unsafe_allow_html=True)
                st.dataframe(percentile_table(df, sel).style.format("{:.4f}"),
                             use_container_width=True)

            if "Skewness & Kurtosis" in analyses:
                st.markdown("<div class='section-header'><h3>Skewness & Kurtosis</h3></div>",
                            unsafe_allow_html=True)
                st.dataframe(skewness_table(df, sel), use_container_width=True)

            if "Normality Test" in analyses:
                st.markdown("<div class='section-header'><h3>Normality — Shapiro-Wilk</h3></div>",
                            unsafe_allow_html=True)
                st.dataframe(normality_test(df, sel), use_container_width=True)

            if "Outlier Summary (IQR)" in analyses:
                st.markdown("<div class='section-header'><h3>Outliers — IQR</h3></div>",
                            unsafe_allow_html=True)
                st.dataframe(outlier_summary_iqr(df, sel), use_container_width=True)

            if "Outlier Summary (Z-Score)" in analyses:
                st.markdown("<div class='section-header'><h3>Outliers — Z-Score</h3></div>",
                            unsafe_allow_html=True)
                st.dataframe(outlier_summary_zscore(df, sel), use_container_width=True)

            if "Variance Analysis" in analyses:
                st.markdown("<div class='section-header'><h3>Variance Analysis</h3></div>",
                            unsafe_allow_html=True)
                st.dataframe(variance_table(df, sel), use_container_width=True)

# ══════════════════════════════════════════════
#  TAB 6 — CATEGORICAL
# ══════════════════════════════════════════════
with TABS[6]:
    df = st.session_state.df
    _, cat_cols, *_ = col_types(df)

    if not cat_cols:
        st.warning("No categorical columns detected.")
    else:
        sel_cat  = st.multiselect("Columns", cat_cols, default=cat_cols[:min(4, len(cat_cols))])
        cat_opts = st.multiselect(
            "Analyses",
            ["Frequency Table", "Cardinality", "Dominant Category", "Rare (<1%)", "Entropy"],
            default=["Frequency Table", "Cardinality"]
        )
        for col in sel_cat:
            st.markdown(f"<div class='section-header'><h3>📌 {col}</h3></div>",
                        unsafe_allow_html=True)
            summary = categorical_summary(df, col)

            if "Frequency Table" in cat_opts:
                st.dataframe(summary["freq_df"], use_container_width=True)
                fig_bar = px.bar(
                    x=summary["freq_df"]["Value"].astype(str).head(20),
                    y=summary["freq_df"]["Count"].head(20),
                    template=template, height=260, title=f"Frequency — {col}",
                    labels={"x": col, "y": "Count"},
                )
                fig_bar.update_layout(font_size=font_sz, showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)

            if "Cardinality" in cat_opts:
                st.metric("Unique Values", summary["cardinality"],
                          delta="High Cardinality ⚠️" if summary["cardinality"] > 50 else "Normal ✅")

            if "Dominant Category" in cat_opts:
                st.info(f"**Dominant:** `{summary['dominant_val']}` — "
                        f"{summary['dominant_count']:,} rows ({summary['dominant_pct']:.1f}%)")

            if "Rare (<1%)" in cat_opts:
                if summary["rare_df"].empty:
                    st.success("No rare categories.")
                else:
                    st.warning(f"{len(summary['rare_df'])} rare categories (<1%)")
                    st.dataframe(summary["rare_df"])

            if "Entropy" in cat_opts:
                st.metric("Shannon Entropy", f"{summary['entropy']:.4f} bits",
                          delta=f"Max possible: {summary['max_entropy']:.2f} bits")

# ══════════════════════════════════════════════
#  TAB 7 — CORRELATION
# ══════════════════════════════════════════════
with TABS[7]:
    df = st.session_state.df
    num_cols, *_ = col_types(df)

    if len(num_cols) < 2:
        st.warning("Need at least 2 numeric columns.")
    else:
        corr_cols = st.multiselect("Columns", num_cols, default=num_cols)
        method    = st.selectbox("Method", ["pearson", "spearman", "kendall"])
        alpha     = st.slider("Significance α", 0.01, 0.10, 0.05, 0.01)

        if len(corr_cols) >= 2:
            cmat = df[corr_cols].corr(method=method)
            fig_heat = px.imshow(
                cmat, text_auto=".2f", color_continuous_scale=palette,
                template=template, height=chart_h,
                title=f"{method.title()} Correlation Heatmap"
            )
            fig_heat.update_layout(font_size=font_sz)
            st.plotly_chart(fig_heat, use_container_width=True)

            # Per-column bar charts
            st.markdown("<div class='section-header'><h3>Per-Column Correlation Bars</h3></div>",
                        unsafe_allow_html=True)
            for ca in corr_cols:
                others = [c for c in corr_cols if c != ca]
                vals   = [round(cmat.loc[ca, c], 4) for c in others]
                colors = ["#dc2626" if v < 0 else "#2563eb" for v in vals]
                fig_b  = go.Figure(go.Bar(
                    x=vals, y=others, orientation="h",
                    marker_color=colors,
                    text=[f"{v:.3f}" for v in vals], textposition="outside"
                ))
                fig_b.update_layout(
                    title=f"Correlation with '{ca}'",
                    xaxis=dict(range=[-1, 1], title="r"),
                    template=template,
                    height=max(260, len(others) * 36 + 80),
                    font_size=font_sz, paper_bgcolor="#fff", plot_bgcolor="#f8f9fc"
                )
                st.plotly_chart(fig_b, use_container_width=True)

            # Ranked pairs
            st.markdown("<div class='section-header'><h3>Ranked Pairs</h3></div>",
                        unsafe_allow_html=True)
            st.dataframe(correlation_table(df, corr_cols, method, alpha),
                         use_container_width=True)

# ══════════════════════════════════════════════
#  TAB 8 — CHARTS
# ══════════════════════════════════════════════
with TABS[8]:
    df = st.session_state.df
    num_cols, cat_cols, *_ = col_types(df)

    chart_type = st.selectbox("Chart Type", [
        "Histogram", "KDE / Density", "Box Plot", "Violin Plot", "Scatter Plot",
        "Bar Chart", "Pie / Donut", "Line Chart", "Area Chart",
        "ECDF Plot", "QQ Plot", "Pair Plot", "Heatmap (Correlation)"
    ])
    col_x   = st.selectbox("X / Primary Column", df.columns.tolist())
    col_y   = st.selectbox("Y Column", ["None"] + df.columns.tolist())
    col_clr = st.selectbox("Color By", ["None"] + cat_cols)
    c_title = st.text_input("Title", chart_type)
    xl, yl  = st.text_input("X Label", col_x), st.text_input("Y Label", col_y if col_y != "None" else "")
    show_leg = st.toggle("Legend", True, key="chart_leg")

    c_arg  = None if col_clr == "None" else col_clr
    y_arg  = None if col_y   == "None" else col_y
    fig_ch = None

    try:
        if chart_type == "Histogram":
            fig_ch = px.histogram(df, x=col_x, color=c_arg,
                                  template=template, height=chart_h, title=c_title, nbins=40)
        elif chart_type == "KDE / Density":
            from scipy.stats import gaussian_kde
            vals = df[col_x].dropna().values; kde = gaussian_kde(vals)
            xs   = np.linspace(vals.min(), vals.max(), 300)
            fig_ch = px.line(x=xs, y=kde(xs), labels={"x": xl, "y": "Density"},
                             template=template, height=chart_h, title=c_title)
        elif chart_type == "Box Plot":
            fig_ch = px.box(df, y=col_x, x=c_arg, color=c_arg,
                            template=template, height=chart_h, title=c_title)
        elif chart_type == "Violin Plot":
            fig_ch = px.violin(df, y=col_x, x=c_arg, color=c_arg,
                               box=True, points="outliers",
                               template=template, height=chart_h, title=c_title)
        elif chart_type == "Scatter Plot" and y_arg:
            fig_ch = px.scatter(df, x=col_x, y=y_arg, color=c_arg,
                                template=template, height=chart_h,
                                title=c_title, trendline="ols")
        elif chart_type == "Bar Chart":
            gb = df[col_x].value_counts().reset_index(); gb.columns = [col_x, "Count"]
            fig_ch = px.bar(gb, x=col_x, y="Count",
                            template=template, height=chart_h, title=c_title)
        elif chart_type == "Pie / Donut":
            vc = df[col_x].value_counts().reset_index(); vc.columns = ["label", "count"]
            fig_ch = px.pie(vc, names="label", values="count", hole=0.4,
                            template=template, height=chart_h, title=c_title)
        elif chart_type == "Line Chart" and y_arg:
            fig_ch = px.line(df, x=col_x, y=y_arg, color=c_arg,
                             template=template, height=chart_h, title=c_title)
        elif chart_type == "Area Chart" and y_arg:
            fig_ch = px.area(df, x=col_x, y=y_arg, color=c_arg,
                             template=template, height=chart_h, title=c_title)
        elif chart_type == "ECDF Plot":
            sv = np.sort(df[col_x].dropna())
            fig_ch = px.line(x=sv, y=np.arange(1, len(sv) + 1) / len(sv),
                             labels={"x": xl, "y": "ECDF"},
                             template=template, height=chart_h, title=c_title)
        elif chart_type == "QQ Plot":
            vals = df[col_x].dropna().values
            (osm, osr), (slope, intercept, _) = stats.probplot(vals)
            fig_ch = go.Figure()
            fig_ch.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name="Data"))
            fig_ch.add_trace(go.Scatter(
                x=osm, y=slope * np.array(osm) + intercept,
                mode="lines", name="Normal", line=dict(color="red", dash="dash")
            ))
            fig_ch.update_layout(template=template, height=chart_h, title=c_title,
                                 xaxis_title="Theoretical Quantiles",
                                 yaxis_title="Sample Quantiles")
        elif chart_type == "Pair Plot":
            pp_cols = st.multiselect("Pair Plot Columns", num_cols,
                                     default=num_cols[:min(5, len(num_cols))])
            if pp_cols:
                fig_ch = px.scatter_matrix(df, dimensions=pp_cols, color=c_arg,
                                           template=template, height=chart_h, title=c_title)
        elif chart_type == "Heatmap (Correlation)":
            cd = df[num_cols].corr()
            fig_ch = px.imshow(cd, text_auto=".2f", color_continuous_scale=palette,
                               template=template, height=chart_h, title=c_title)

        if fig_ch:
            fig_ch.update_layout(
                font_size=font_sz, showlegend=show_leg,
                xaxis_title=xl, yaxis_title=yl,
                xaxis=dict(showgrid=show_grid), yaxis=dict(showgrid=show_grid)
            )
            st.plotly_chart(fig_ch, use_container_width=True)
            fmt = st.selectbox("Export chart as", ["png", "svg", "html"], key="chart_fmt")
            if st.button("⬇️ Download Chart"):
                if fmt == "html":
                    buf = io.StringIO(); fig_ch.write_html(buf)
                    st.download_button("Download HTML", buf.getvalue(), f"{c_title}.html")
                else:
                    st.download_button(f"Download {fmt.upper()}",
                                       fig_ch.to_image(format=fmt),
                                       f"{c_title}.{fmt}", f"image/{fmt}")
        else:
            st.info("Choose valid columns for this chart type.")
    except Exception as e:
        st.error(f"Chart error: {e}")

# ══════════════════════════════════════════════
#  TAB 9 — TRANSFORM
# ══════════════════════════════════════════════
with TABS[9]:
    df = st.session_state.df
    num_cols, *_ = col_types(df)

    if not num_cols:
        st.warning("No numeric columns.")
    else:
        st.markdown("<div class='section-header'><h3>Apply Transformations</h3></div>",
                    unsafe_allow_html=True)
        t_col   = st.selectbox("Column", num_cols, key="t_col_tr")
        t_label = st.text_input("New column name (blank = auto)", "", key="t_label")
        t_map   = {
            "log":       "Log (log1p)",
            "sqrt":      "Square Root",
            "boxcox":    "Box-Cox",
            "yeo":       "Yeo-Johnson",
            "standard":  "Standard Scaling (Z-score)",
            "minmax":    "MinMax Scaling [0,1]",
            "robust":    "Robust Scaling",
            "square":    "Square",
            "cbrt":      "Cube Root",
            "reciprocal":"Reciprocal (1/x)",
        }
        t_opts = st.multiselect(
            "Transformations", list(t_map.keys()),
            format_func=lambda x: t_map[x]
        )

        if st.button("✅ Apply") and t_opts:
            d = st.session_state.df.copy()
            results = {}
            for t in t_opts:
                lab = t_label if (t_label and len(t_opts) == 1) else None
                try:
                    d, new_col = apply_transform(d, t_col, t, label=lab)
                    results[new_col] = d[new_col].describe().round(4)
                    slog(f"Added transform '{new_col}' ({t_map[t]} of '{t_col}')")
                except Exception as e:
                    st.warning(f"Could not apply '{t_map[t]}': {e}")
            if results:
                st.session_state.df = d
                st.success(f"Added {len(results)} column(s).")
                st.dataframe(pd.DataFrame(results).T.style.format("{:.4f}"),
                             use_container_width=True)

# ══════════════════════════════════════════════
#  TAB 10 — RECOMMENDATIONS
# ══════════════════════════════════════════════
with TABS[10]:
    df = st.session_state.df

    st.markdown("<div class='section-header'><h3>Automated Recommendations</h3></div>",
                unsafe_allow_html=True)
    st.caption("Scans your dataset and provides step-by-step, actionable guidance with Python code.")

    recs = generate_recommendations(df)
    score_r, grade_r, n_crit, n_warn = recommendations_score(recs)

    level_cls = {"critical": "rec-danger", "warning": "rec-warn",
                 "ok": "rec-ok", "info": "rec-info"}
    level_ico = {"critical": "🔴", "warning": "🟡", "ok": "🟢", "info": "🔵"}

    # Group by topic headings
    topics = [
        ("1️⃣ Duplicate Rows",          lambda r: "duplicate" in r["title"].lower()),
        ("2️⃣ Missing Values",          lambda r: "missing" in r["title"].lower()),
        ("3️⃣ Skewness",               lambda r: "skew" in r["title"].lower()),
        ("4️⃣ Outliers",               lambda r: "outlier" in r["title"].lower() or "iqr" in r["title"].lower()),
        ("5️⃣ Cardinality",            lambda r: "cardinality" in r["title"].lower() or "unique values" in r["title"].lower() or "constant" in r["title"].lower()),
        ("6️⃣ Multicollinearity",      lambda r: "correlat" in r["title"].lower()),
        ("7️⃣ Class Distribution",     lambda r: "dominat" in r["title"].lower() or "imbalance" in r["title"].lower()),
    ]

    for heading, matcher in topics:
        matching = [r for r in recs if matcher(r)]
        if matching:
            st.markdown(f"<div class='section-header'><h3>{heading}</h3></div>",
                        unsafe_allow_html=True)
            for r in matching:
                cls = level_cls.get(r["level"], "rec-info")
                ico = level_ico.get(r["level"], "ℹ️")
                st.markdown(
                    f"<div class='{cls}'><b>{ico} {r['title']}</b><br>{r['body']}</div>",
                    unsafe_allow_html=True
                )

    # Scorecard
    st.markdown("<div class='section-header'><h3>📊 Quality Scorecard</h3></div>",
                unsafe_allow_html=True)
    cs1, cs2, cs3, cs4 = st.columns(4)
    for w, lb, v in zip([cs1, cs2, cs3, cs4],
                        ["Quality Score", "Grade", "Critical Issues", "Warnings"],
                        [f"{score_r}/100", grade_r, n_crit, n_warn]):
        with w: st.metric(lb, v)

    if   score_r == 100: st.success("🎉 Dataset is in excellent shape!")
    elif score_r >= 75:  st.info(f"📈 Good quality. Address {n_crit + n_warn} issue(s) before modelling.")
    else:                st.warning("⚠️ Several issues need attention before analysis.")

    # EDA Checklist
    st.markdown("<div class='section-header'><h3>✅ EDA Checklist</h3></div>",
                unsafe_allow_html=True)
    num_cols_c, cat_cols_c, *_ = col_types(df)
    all_stats_c = {c: full_stats(df[c]) for c in num_cols_c}
    miss_pct_c  = df.isnull().mean() * 100
    n_dup_c     = df.duplicated().sum()
    checks = [
        ("Dataset shape & dtypes reviewed",                                          True),
        ("Missing values handled",                          not (miss_pct_c > 0).any()),
        ("Duplicate rows removed",                                        n_dup_c == 0),
        ("Outliers detected & treated",
         all(all_stats_c.get(c, {}).get("out_iqr", 0) == 0 for c in num_cols_c)),
        ("Skewness checked & corrected",
         all(abs(all_stats_c.get(c, {}).get("skew", 0)) < 1 for c in num_cols_c)),
        ("Normality verified (Shapiro-Wilk / Q-Q plot)",
         all(all_stats_c.get(c, {}).get("is_normal", True) for c in num_cols_c)),
        ("Categorical columns reviewed",                                             True),
        ("Multicollinearity checked",                                                True),
        ("Features scaled / normalised (if needed)",                                False),
        ("Class balance verified",                                                   True),
        ("Cleaned dataset exported",                                                False),
    ]
    for item, done in checks:
        st.markdown(f"{'✅' if done else '⬜'} {item}")

# ══════════════════════════════════════════════
#  TAB 11 — EXPORT & HISTORY
# ══════════════════════════════════════════════
with TABS[11]:
    df = st.session_state.df
    num_cols, *_ = col_types(df)

    st.markdown("<div class='section-header'><h3>Current Dataset</h3></div>",
                unsafe_allow_html=True)
    st.info(
        f"**{len(df):,} rows × {df.shape[1]} columns** "
        f"| Missing: **{df.isnull().sum().sum():,}** "
        f"| Duplicates: **{df.duplicated().sum()}** "
        f"| File: `{st.session_state.fname}`"
    )

    st.markdown("<div class='section-header'><h3>⬇️ Download Cleaned Dataset</h3></div>",
                unsafe_allow_html=True)
    c1e, c2e, c3e = st.columns(3)
    with c1e:
        st.download_button("⬇️ CSV",   to_csv_bytes(df),
                           "cleaned_data.csv",  "text/csv", use_container_width=True)
    with c2e:
        st.download_button("⬇️ Excel", to_excel_bytes(df),
                           "cleaned_data.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)
    with c3e:
        st.download_button("⬇️ JSON",  to_json_bytes(df),
                           "cleaned_data.json", "application/json", use_container_width=True)

    if num_cols:
        st.markdown("<div class='section-header'><h3>Summary Statistics</h3></div>",
                    unsafe_allow_html=True)
        summ = descriptive_stats(df, num_cols)
        st.dataframe(summ.style.format("{:.4f}"), use_container_width=True)
        st.download_button("⬇️ Summary Stats CSV",
                           summ.to_csv().encode(), "summary_stats.csv", "text/csv")

    # Quality Gauge
    st.markdown("<div class='section-header'><h3>📊 Data Quality Gauge</h3></div>",
                unsafe_allow_html=True)
    qs = quality_score(df)
    qc = "#16a34a" if qs >= 80 else "#d97706" if qs >= 60 else "#dc2626"
    try:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=qs,
            title={"text": "Quality Score", "font": {"size": 18}},
            number={"font": {"color": qc, "size": 48}},
            gauge={
                "axis":      {"range": [0, 100]},
                "bar":       {"color": qc},
                "steps":     [{"range": [0,  40], "color": "#fee2e2"},
                              {"range": [40, 70], "color": "#fef9c3"},
                              {"range": [70,100], "color": "#dcfce7"}],
                "threshold": {"line": {"color": "#374151", "width": 3},
                              "thickness": 0.75, "value": qs}
            }
        ))
        fig_g.update_layout(template="plotly_white", height=300, paper_bgcolor="#ffffff")
        st.plotly_chart(fig_g, use_container_width=True)
    except Exception as e:
        st.error(f"Gauge error: {e}")

    # Score breakdown bars
    st.markdown("<div class='section-header'><h3>Score Breakdown</h3></div>",
                unsafe_allow_html=True)
    bd = quality_breakdown(df)
    for factor, (pts, mx) in bd.items():
        pct = pts / mx * 100
        bc  = "#16a34a" if pct >= 80 else "#d97706" if pct >= 60 else "#dc2626"
        st.markdown(f"""
        <div style='margin:8px 0;padding:12px 16px;background:#fff;border:1px solid #e2e6f0;border-radius:10px;'>
          <div style='display:flex;justify-content:space-between;margin-bottom:5px;'>
            <span style='font-size:.85rem;color:#374151;font-weight:500;'>{factor}</span>
            <span style='font-family:"JetBrains Mono",monospace;font-size:.85rem;color:{bc};font-weight:700;'>{pts}/{mx}</span>
          </div>
          <div style='background:#f1f3f9;border-radius:6px;height:6px;'>
            <div style='width:{pct:.1f}%;height:100%;background:{bc};border-radius:6px;'></div>
          </div>
        </div>""", unsafe_allow_html=True)

    # Processing history
    st.markdown("<div class='section-header'><h3>📜 Processing History</h3></div>",
                unsafe_allow_html=True)
    hist = get_history(st.session_state.fname)
    if hist.empty:
        st.info("No operations recorded yet.")
    else:
        st.dataframe(
            hist.rename(columns={"ts": "Timestamp", "operation": "Operation", "details": "Details"}),
            use_container_width=True, height=300
        )
        st.download_button("⬇️ Export History CSV",
                           hist.to_csv(index=False).encode(),
                           "processing_history.csv", "text/csv")

    # Session log
    st.markdown("<div class='section-header'><h3>Session Log</h3></div>",
                unsafe_allow_html=True)
    log_text = "\n".join(st.session_state.log) if st.session_state.log else "No operations yet."
    st.text_area("All steps this session", log_text, height=160)
    if st.session_state.log:
        st.download_button("⬇️ Download Log (.txt)",
                           log_text.encode(), "session_log.txt", "text/plain")
