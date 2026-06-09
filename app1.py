# ============================================================
#  app.py — Descriptive Analytics Workbench  (Enhanced)
#  NEW: Duplicate removal, outlier filling (mean/median/mode/cap),
#       full Recommendations tab with actionable steps
#  Run: streamlit run app.py
# ============================================================
import io
import pickle
import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Analytics Workbench",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .block-container { padding: 1.5rem 2rem; }
    .stTabs [data-baseweb="tab"] { font-size: 0.85rem; font-weight: 600; letter-spacing: 0.04em; }
    .metric-card {
        background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;
        padding: 1rem 1.2rem; margin-bottom: 0.5rem;
    }
    .section-header {
        font-size: 1.05rem; font-weight: 700; color: #1e293b;
        border-left: 4px solid #6366f1; padding-left: 0.7rem; margin: 1.2rem 0 0.8rem;
    }
    .rec-card {
        background: linear-gradient(135deg,#f0f9ff,#e0f2fe);
        border: 1px solid #bae6fd; border-radius: 10px;
        padding: 1rem 1.2rem; margin-bottom: 0.6rem;
    }
    .rec-card-warn {
        background: linear-gradient(135deg,#fffbeb,#fef3c7);
        border: 1px solid #fde68a; border-radius: 10px;
        padding: 1rem 1.2rem; margin-bottom: 0.6rem;
    }
    .rec-card-danger {
        background: linear-gradient(135deg,#fff1f2,#ffe4e6);
        border: 1px solid #fecdd3; border-radius: 10px;
        padding: 1rem 1.2rem; margin-bottom: 0.6rem;
    }
    .rec-card-ok {
        background: linear-gradient(135deg,#f0fdf4,#dcfce7);
        border: 1px solid #bbf7d0; border-radius: 10px;
        padding: 1rem 1.2rem; margin-bottom: 0.6rem;
    }
    code { font-family: 'DM Mono', monospace; background:#f1f5f9; padding:2px 5px; border-radius:4px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SESSION STATE  (so cleaning persists)
# ─────────────────────────────────────────────
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None
if "clean_log" not in st.session_state:
    st.session_state.clean_log = []


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        xl = pd.ExcelFile(uploaded_file)
        return xl, xl.sheet_names
    return None

def detect_col_types(df):
    num_cols  = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols  = df.select_dtypes(include=["object", "category"]).columns.tolist()
    date_cols = df.select_dtypes(include=["datetime"]).columns.tolist()
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    return num_cols, cat_cols, date_cols, bool_cols

def df_to_excel_bytes(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False)
    buf.seek(0)
    return buf

def iqr_fences(s):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5*iqr, q3 + 1.5*iqr, iqr

def numeric_full_stats(s):
    s = s.dropna()
    if len(s) == 0:
        return {}
    lo, hi, iqr = iqr_fences(s)
    if 3 <= len(s) <= 5000:
        _, p_norm = stats.shapiro(s); norm_test = "Shapiro-Wilk"
    else:
        _, p_norm = stats.normaltest(s); norm_test = "D'Agostino"
    skew = s.skew()
    kurt = s.kurtosis()
    return {
        "count": len(s), "mean": s.mean(), "median": s.median(),
        "mode": s.mode().iloc[0] if not s.mode().empty else None,
        "std": s.std(), "variance": s.var(),
        "min": s.min(), "max": s.max(), "range": s.max()-s.min(),
        "q1": s.quantile(0.25), "q3": s.quantile(0.75), "iqr": iqr,
        "p5": s.quantile(0.05), "p95": s.quantile(0.95),
        "skewness": round(skew, 4), "kurtosis": round(kurt, 4),
        "cv_pct": round(s.std()/s.mean()*100, 2) if s.mean() != 0 else None,
        "norm_test": norm_test, "norm_p": round(float(p_norm), 5),
        "is_normal": p_norm >= 0.05,
        "outliers_iqr": int(((s < lo) | (s > hi)).sum()),
        "outliers_z": int((np.abs(stats.zscore(s)) > 3).sum()),
        "lower_fence": round(lo, 4), "upper_fence": round(hi, 4),
    }


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📂 Dataset")
    upload_mode = st.radio("Mode", ["Single File", "Multiple Files (Merge)"], horizontal=True)

    df_raw = None

    if upload_mode == "Single File":
        uploaded = st.file_uploader("Upload CSV / Excel", type=["csv", "xlsx", "xls"])
        if uploaded:
            result = load_file(uploaded)
            if isinstance(result, tuple):
                xl, sheet_names = result
                sheet = st.selectbox("Select Sheet", sheet_names)
                df_raw = xl.parse(sheet)
            else:
                df_raw = result
    else:
        files = st.file_uploader("Upload Multiple CSVs/Excels", type=["csv","xlsx","xls"],
                                  accept_multiple_files=True)
        if files:
            frames = []
            for f in files:
                r = load_file(f)
                frames.append(r if not isinstance(r, tuple) else r[0].parse(r[1][0]))
            merge_how = st.selectbox("Merge Strategy", ["Vertical (concat)", "Horizontal (concat)"])
            try:
                df_raw = pd.concat(frames, axis=0 if "Vertical" in merge_how else 1, ignore_index=True)
                st.success(f"Merged {len(files)} files → {df_raw.shape}")
            except Exception as e:
                st.error(f"Merge failed: {e}")

    if df_raw is not None:
        # If a new file is uploaded, reset cleaning state
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
                df_work.rename(columns=renames, inplace=True)

        dtype_mode = st.checkbox("Change Data Types")
        if dtype_mode:
            for c in df_work.columns:
                cur = str(df_work[c].dtype)
                new_dtype = st.selectbox(
                    f"'{c}' ({cur})", ["(keep)","numeric","string","category","datetime"],
                    key=f"dt_{c}"
                )
                if new_dtype == "numeric":
                    df_work[c] = pd.to_numeric(df_work[c], errors="coerce")
                elif new_dtype == "string":
                    df_work[c] = df_work[c].astype(str)
                elif new_dtype == "category":
                    df_work[c] = df_work[c].astype("category")
                elif new_dtype == "datetime":
                    df_work[c] = pd.to_datetime(df_work[c], errors="coerce")

        st.markdown("---")
        st.markdown("### 🔍 Row Filters")
        filter_query = st.text_area(
            "Custom Filter (pandas query)",
            placeholder='e.g.  Age > 25 and Income < 100000',
            height=80
        )
        if filter_query:
            try:
                df_work = df_work.query(filter_query)
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


# ─────────────────────────────────────────────
#  MAIN AREA
# ─────────────────────────────────────────────

st.title("📊 Descriptive Analytics Workbench")
st.caption("Upload your dataset → explore → clean → transform → export → stay ML-ready.")

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
    "🧹 Data Cleaning",       # ← ENHANCED tab (was ML Ready)
    "💾 Export",
    "💡 Recommendations",     # ← NEW full tab
])


# ══════════════════════════════════════════════
#  TAB 1 — OVERVIEW
# ══════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-header">Dataset Snapshot</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows",        f"{df_work.shape[0]:,}")
    c2.metric("Columns",     f"{df_work.shape[1]:,}")
    c3.metric("Numeric",     len(num_cols))
    c4.metric("Categorical", len(cat_cols))

    st.dataframe(df_work.head(50), use_container_width=True)

    st.markdown('<div class="section-header">Data Types & Quality</div>', unsafe_allow_html=True)
    dtype_df = pd.DataFrame({
        "Column":        df_work.columns,
        "Dtype":         df_work.dtypes.astype(str).values,
        "Non-Null":      df_work.notna().sum().values,
        "Null Count":    df_work.isna().sum().values,
        "Null %":        (df_work.isna().mean()*100).round(2).values,
        "Unique Values": df_work.nunique().values,
    })
    st.dataframe(dtype_df, use_container_width=True)

    st.markdown('<div class="section-header">Missing Values</div>', unsafe_allow_html=True)
    miss = df_work.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if miss.empty:
        st.success("✅ No missing values found!")
    else:
        fig = px.bar(x=miss.index, y=miss.values,
                     labels={"x":"Column","y":"Missing Count"},
                     color=miss.values, color_continuous_scale=palette,
                     template=template, height=chart_h//1.5)
        fig.update_layout(font_size=font_sz, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Duplicate Analysis</div>', unsafe_allow_html=True)
    dupes = df_work.duplicated().sum()
    st.metric("Duplicate Rows", dupes,
              delta=f"{dupes/len(df_work)*100:.1f}% of data" if dupes else "Clean ✅")
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
            "Descriptive Statistics","Percentile Table","Skewness & Kurtosis",
            "Normality Test (Shapiro-Wilk)","Outlier Detection (IQR)",
            "Outlier Detection (Z-Score)","Variance Analysis"
        ], default=["Descriptive Statistics"])

        if sel_num:
            df_num = df_work[sel_num]

            if "Descriptive Statistics" in analyses:
                st.markdown('<div class="section-header">Descriptive Statistics</div>', unsafe_allow_html=True)
                desc = df_num.describe().T
                desc["cv%"]   = (desc["std"] / desc["mean"] * 100).round(2)
                desc["range"] = desc["max"] - desc["min"]
                desc["iqr"]   = df_num.quantile(0.75) - df_num.quantile(0.25)
                desc["mad"]   = df_num.apply(lambda x: (x-x.mean()).abs().mean())
                st.dataframe(desc.style.format("{:.4f}"), use_container_width=True)

            if "Percentile Table" in analyses:
                st.markdown('<div class="section-header">Percentile Table</div>', unsafe_allow_html=True)
                pcts = [1,5,10,25,50,75,90,95,99]
                pct_df = df_num.quantile([p/100 for p in pcts]).T
                pct_df.columns = [f"P{p}" for p in pcts]
                st.dataframe(pct_df.style.format("{:.4f}"), use_container_width=True)

            if "Skewness & Kurtosis" in analyses:
                st.markdown('<div class="section-header">Skewness & Kurtosis</div>', unsafe_allow_html=True)
                sk_df = pd.DataFrame({
                    "Skewness":  df_num.skew().round(4),
                    "Kurtosis":  df_num.kurtosis().round(4),
                    "Skew Interpretation": df_num.skew().apply(
                        lambda x: "Highly Negative" if x < -1
                        else "Moderate Negative" if x < -0.5
                        else "Approx. Normal" if abs(x) <= 0.5
                        else "Moderate Positive" if x < 1
                        else "Highly Positive"
                    )
                })
                st.dataframe(sk_df, use_container_width=True)

            if "Normality Test (Shapiro-Wilk)" in analyses:
                st.markdown('<div class="section-header">Normality Test</div>', unsafe_allow_html=True)
                norm_rows = []
                for col in sel_num:
                    sample = df_num[col].dropna()
                    if len(sample) > 5000:
                        sample = sample.sample(5000, random_state=42)
                    stat_v, p = stats.shapiro(sample)
                    norm_rows.append({
                        "Column": col, "Statistic": round(stat_v,4), "p-value": round(p,6),
                        "Normal?": "✅ Yes (p>0.05)" if p > 0.05 else "❌ No (p≤0.05)"
                    })
                st.dataframe(pd.DataFrame(norm_rows), use_container_width=True)

            if "Outlier Detection (IQR)" in analyses:
                st.markdown('<div class="section-header">Outlier Detection — IQR</div>', unsafe_allow_html=True)
                iqr_rows = []
                for col in sel_num:
                    s = df_num[col].dropna()
                    lo, hi, iqr_v = iqr_fences(s)
                    n_out = ((s < lo) | (s > hi)).sum()
                    iqr_rows.append({
                        "Column": col, "Q1": round(s.quantile(0.25),4),
                        "Q3": round(s.quantile(0.75),4), "IQR": round(iqr_v,4),
                        "Lower Fence": round(lo,4), "Upper Fence": round(hi,4),
                        "Outlier Count": n_out,
                        "Outlier %": round(n_out/len(s)*100, 2)
                    })
                st.dataframe(pd.DataFrame(iqr_rows), use_container_width=True)

            if "Outlier Detection (Z-Score)" in analyses:
                st.markdown('<div class="section-header">Outlier Detection — Z-Score</div>', unsafe_allow_html=True)
                z_rows = []
                for col in sel_num:
                    s = df_num[col].dropna()
                    z = np.abs(stats.zscore(s))
                    n_out = (z > 3).sum()
                    z_rows.append({
                        "Column": col, "Mean": round(s.mean(),4), "Std": round(s.std(),4),
                        "Outlier Count (|z|>3)": n_out,
                        "Outlier %": round(n_out/len(s)*100, 2)
                    })
                st.dataframe(pd.DataFrame(z_rows), use_container_width=True)

            if "Variance Analysis" in analyses:
                st.markdown('<div class="section-header">Variance Analysis</div>', unsafe_allow_html=True)
                var_df = pd.DataFrame({
                    "Variance": df_num.var().round(4),
                    "Std Dev":  df_num.std().round(4),
                    "CV %":     (df_num.std()/df_num.mean()*100).round(2)
                })
                st.dataframe(var_df, use_container_width=True)


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
            "Frequency Table","Cardinality","Dominant Category",
            "Rare Categories (<1%)","Entropy Analysis"
        ], default=["Frequency Table","Cardinality"])

        for col in sel_cat:
            st.markdown(f'<div class="section-header">📌 {col}</div>', unsafe_allow_html=True)
            s = df_work[col].dropna()
            vc  = s.value_counts()
            vcp = s.value_counts(normalize=True) * 100

            if "Frequency Table" in cat_analyses:
                freq_df = pd.DataFrame({
                    "Value": vc.index, "Count": vc.values,
                    "Percent %": vcp.values.round(2)
                })
                st.dataframe(freq_df, use_container_width=True)

            if "Cardinality" in cat_analyses:
                st.metric(f"Unique Values in '{col}'", s.nunique(),
                          delta="High Cardinality ⚠️" if s.nunique() > 50 else "Normal ✅")

            if "Dominant Category" in cat_analyses:
                dom = vc.idxmax()
                st.info(f"**Dominant:** `{dom}` → {vc.max():,} rows ({vcp.max():.1f}%)")

            if "Rare Categories (<1%)" in cat_analyses:
                rare = vcp[vcp < 1]
                if rare.empty:
                    st.success("No rare categories found.")
                else:
                    st.warning(f"{len(rare)} rare categories (<1%):")
                    st.dataframe(pd.DataFrame({"Category": rare.index, "Percent %": rare.values.round(3)}))

            if "Entropy Analysis" in cat_analyses:
                from scipy.stats import entropy
                probs = vc / vc.sum()
                ent = entropy(probs, base=2)
                max_ent = np.log2(s.nunique()) if s.nunique() > 1 else 1
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
        corr_method = st.selectbox("Method", ["pearson","spearman","kendall"])
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
            pairs = []
            for i in range(len(corr_df.columns)):
                for j in range(i+1, len(corr_df.columns)):
                    c1n, c2n = corr_df.columns[i], corr_df.columns[j]
                    r = corr_df.loc[c1n, c2n]
                    n = df_work[[c1n,c2n]].dropna().shape[0]
                    t_stat = r * np.sqrt(n-2) / np.sqrt(1-r**2+1e-10)
                    p_val  = 2 * stats.t.sf(abs(t_stat), df=n-2)
                    pairs.append({
                        "Col A": c1n, "Col B": c2n,
                        "Correlation": round(r, 4),
                        "p-value": round(p_val, 6),
                        "Significant": "✅" if p_val < sig_level else "❌",
                        "Strength": "Strong" if abs(r)>0.7 else "Moderate" if abs(r)>0.4 else "Weak"
                    })
            pairs_df = pd.DataFrame(pairs).sort_values("Correlation", key=abs, ascending=False)
            st.dataframe(pairs_df, use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 5 — CHARTS
# ══════════════════════════════════════════════
with tabs[4]:
    chart_type = st.selectbox("Chart Type", [
        "Histogram","KDE / Density","Box Plot","Violin Plot",
        "Scatter Plot","Bar Chart","Pie / Donut Chart",
        "Line Chart","Area Chart","Pair Plot",
        "ECDF Plot","QQ Plot","Heatmap (Correlation)"
    ])
    col_x      = st.selectbox("X / Primary Column", df_work.columns.tolist())
    col_y      = st.selectbox("Y Column (if needed)", ["None"] + df_work.columns.tolist())
    col_color  = st.selectbox("Color By (optional)", ["None"] + cat_cols)
    chart_title   = st.text_input("Chart Title", value=chart_type)
    x_label       = st.text_input("X-Axis Label",  value=col_x)
    y_label       = st.text_input("Y-Axis Label",  value=col_y if col_y != "None" else "")
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
            fig_chart = px.line(x=xs, y=kde(xs), labels={"x":x_label,"y":"Density"},
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
            vc.columns = ["label","count"]
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
            ecdf = np.arange(1, len(sorted_vals)+1) / len(sorted_vals)
            fig_chart = px.line(x=sorted_vals, y=ecdf, labels={"x":x_label,"y":"ECDF"},
                template=template, height=chart_h, title=chart_title)

        elif chart_type == "QQ Plot":
            vals = df_work[col_x].dropna().values
            (osm, osr), (slope, intercept, r) = stats.probplot(vals, dist="norm")
            fig_chart = go.Figure()
            fig_chart.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name="Data"))
            fig_chart.add_trace(go.Scatter(x=osm, y=slope*np.array(osm)+intercept,
                mode="lines", name="Normal Line", line=dict(color="red",dash="dash")))
            fig_chart.update_layout(template=template, height=chart_h, title=chart_title,
                xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles")

        elif chart_type == "Pair Plot":
            sel_pp = st.multiselect("Choose Columns for Pair Plot", num_cols,
                                     default=num_cols[:min(5,len(num_cols))])
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
            fmt = st.selectbox("Export Chart As", ["png","svg","html"])
            if st.button("⬇️ Download Chart"):
                if fmt == "html":
                    buf = io.StringIO()
                    fig_chart.write_html(buf)
                    st.download_button("Download HTML", buf.getvalue(), file_name=f"{chart_title}.html")
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
            "Log (log1p)","Square Root","Box-Cox","Yeo-Johnson",
            "Standard Scaling (Z-score)","MinMax Scaling [0,1]","Robust Scaling"
        ])
        df_transformed = df_work.copy()

        if t_col and transforms:
            from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
            col_data = df_transformed[[t_col]].dropna()
            preview_rows = {}
            for t in transforms:
                try:
                    if t == "Log (log1p)":
                        transformed = np.log1p(col_data[t_col]); label = f"{t_col}_log1p"
                    elif t == "Square Root":
                        transformed = np.sqrt(col_data[t_col].clip(lower=0)); label = f"{t_col}_sqrt"
                    elif t == "Box-Cox":
                        pt = PowerTransformer(method="box-cox")
                        transformed = pt.fit_transform(col_data[[t_col]]).flatten(); label = f"{t_col}_boxcox"
                    elif t == "Yeo-Johnson":
                        pt = PowerTransformer(method="yeo-johnson")
                        transformed = pt.fit_transform(col_data[[t_col]]).flatten(); label = f"{t_col}_yeojohnson"
                    elif t == "Standard Scaling (Z-score)":
                        sc = StandardScaler()
                        transformed = sc.fit_transform(col_data[[t_col]]).flatten(); label = f"{t_col}_zscore"
                    elif t == "MinMax Scaling [0,1]":
                        sc = MinMaxScaler()
                        transformed = sc.fit_transform(col_data[[t_col]]).flatten(); label = f"{t_col}_minmax"
                    elif t == "Robust Scaling":
                        sc = RobustScaler()
                        transformed = sc.fit_transform(col_data[[t_col]]).flatten(); label = f"{t_col}_robust"
                    df_transformed.loc[col_data.index, label] = transformed
                    preview_rows[label] = pd.Series(transformed).describe().round(4)
                except Exception as e:
                    st.warning(f"Could not apply '{t}': {e}")
            st.dataframe(pd.DataFrame(preview_rows).T, use_container_width=True)
            st.success("Transformations applied. Download from Export tab.")


# ══════════════════════════════════════════════
#  TAB 7 — DATA CLEANING  (NEW ENHANCED TAB)
# ══════════════════════════════════════════════
with tabs[6]:
    st.markdown("### 🧹 Data Cleaning Operations")
    st.caption("All operations modify the working dataset and are logged below.")

    # Work with session state df
    df_clean_work = st.session_state.df_clean[keep].copy()
    num_c, cat_c, _, _ = detect_col_types(df_clean_work)

    # ── SECTION A: DUPLICATE REMOVAL ──────────────────────────
    st.markdown('<div class="section-header">🔁 Duplicate Row Removal</div>', unsafe_allow_html=True)

    dup_count = df_clean_work.duplicated().sum()
    col_a, col_b = st.columns([2,1])
    with col_a:
        dup_subset = st.multiselect(
            "Check duplicates based on columns (leave empty = all columns)",
            df_clean_work.columns.tolist(), key="dup_subset"
        )
        dup_keep = st.selectbox("Which duplicate to keep?",
            ["first", "last", "none (drop all)"], key="dup_keep")
    with col_b:
        st.metric("Current Duplicate Rows", dup_count,
                  delta="✅ No duplicates" if dup_count == 0 else f"⚠️ {dup_count} found")

    if st.button("🗑️ Remove Duplicates", disabled=(dup_count == 0), key="btn_dup"):
        subset_arg = dup_subset if dup_subset else None
        keep_arg   = False if "none" in dup_keep else dup_keep
        before = len(st.session_state.df_clean)
        st.session_state.df_clean.drop_duplicates(
            subset=subset_arg, keep=keep_arg, inplace=True
        )
        st.session_state.df_clean.reset_index(drop=True, inplace=True)
        after = len(st.session_state.df_clean)
        removed = before - after
        st.session_state.clean_log.append(f"✅ Removed {removed} duplicate rows (keep='{dup_keep}')")
        st.success(f"Removed {removed} duplicate rows. Dataset now has {after:,} rows.")
        st.rerun()

    st.divider()

    # ── SECTION B: MISSING VALUE TREATMENT ───────────────────
    st.markdown('<div class="section-header">🕳️ Missing Value Imputation</div>', unsafe_allow_html=True)

    miss_df = df_clean_work.isna().sum()
    miss_df = miss_df[miss_df > 0]

    if miss_df.empty:
        st.success("✅ No missing values in the dataset!")
    else:
        st.info(f"**{miss_df.sum():,} missing values** across **{len(miss_df)} columns**")
        st.dataframe(pd.DataFrame({
            "Column": miss_df.index,
            "Missing Count": miss_df.values,
            "Missing %": (miss_df / len(df_clean_work) * 100).round(2).values
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
                        key=f"miss_{col}"
                    )
                    const_val = None
                    if method == "Fill with Constant":
                        const_val = st.number_input(f"Constant for '{col}'", key=f"const_{col}", value=0.0)
                else:
                    method = st.selectbox(
                        f"Strategy for '{col}'",
                        ["(skip)", "Fill with Mode", "Fill with Constant",
                         "Fill with Forward Fill (ffill)", "Fill with Backward Fill (bfill)",
                         "Drop rows with missing"],
                        key=f"miss_{col}"
                    )
                    const_val = None
                    if method == "Fill with Constant":
                        const_val = st.text_input(f"Constant for '{col}'", key=f"const_{col}", value="Unknown")
                impute_actions[col] = (method, const_val)

        if st.button("✅ Apply Imputation", key="btn_impute"):
            df_imp = st.session_state.df_clean.copy()
            applied = []
            for col, (method, const_val) in impute_actions.items():
                if method == "(skip)": continue
                if col not in df_imp.columns: continue
                if method == "Fill with Mean":
                    df_imp[col].fillna(df_imp[col].mean(), inplace=True)
                elif method == "Fill with Median":
                    df_imp[col].fillna(df_imp[col].median(), inplace=True)
                elif method == "Fill with Mode":
                    df_imp[col].fillna(df_imp[col].mode().iloc[0], inplace=True)
                elif method == "Fill with Constant":
                    df_imp[col].fillna(const_val, inplace=True)
                elif method == "Fill with Forward Fill (ffill)":
                    df_imp[col].fillna(method="ffill", inplace=True)
                elif method == "Fill with Backward Fill (bfill)":
                    df_imp[col].fillna(method="bfill", inplace=True)
                elif method == "Drop rows with missing":
                    df_imp.dropna(subset=[col], inplace=True)
                    df_imp.reset_index(drop=True, inplace=True)
                applied.append(f"{col} → {method}")
            st.session_state.df_clean = df_imp
            for a in applied:
                st.session_state.clean_log.append(f"✅ Imputed: {a}")
            st.success(f"Applied imputation to {len(applied)} column(s).")
            st.rerun()

    st.divider()

    # ── SECTION C: OUTLIER TREATMENT ──────────────────────────
    st.markdown('<div class="section-header">📌 Outlier Treatment</div>', unsafe_allow_html=True)

    if not num_c:
        st.info("No numeric columns available for outlier treatment.")
    else:
        out_col = st.selectbox("Select Numeric Column", num_c, key="out_col")
        s_out   = df_clean_work[out_col].dropna()
        lo, hi, iqr_v = iqr_fences(s_out)
        n_out_iqr = ((s_out < lo) | (s_out > hi)).sum()
        n_out_z   = (np.abs(stats.zscore(s_out)) > 3).sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("IQR Outliers",      n_out_iqr)
        c2.metric("Z-Score Outliers",  n_out_z)
        c3.metric("Lower / Upper Fence", f"{lo:.2f} / {hi:.2f}")

        out_method = st.selectbox("Detection Method", ["IQR (1.5×IQR)", "Z-Score (|z| > 3)"], key="out_method")
        out_action = st.selectbox("Treatment Action", [
            "Cap / Winsorise (clip to fences)",
            "Fill with Mean",
            "Fill with Median",
            "Fill with Mode",
            "Remove rows with outliers",
        ], key="out_action")

        if st.button(f"⚡ Apply Outlier Treatment to '{out_col}'", key="btn_out"):
            df_out = st.session_state.df_clean.copy()
            s_col  = df_out[out_col].dropna()

            if "IQR" in out_method:
                lo_f, hi_f, _ = iqr_fences(s_col)
                mask = (df_out[out_col] < lo_f) | (df_out[out_col] > hi_f)
            else:  # Z-Score
                z_scores = np.abs(stats.zscore(df_out[out_col].dropna()))
                z_idx    = df_out[out_col].dropna().index
                mask     = pd.Series(False, index=df_out.index)
                mask.loc[z_idx] = (z_scores > 3)
                lo_f, hi_f = df_out[out_col].mean() - 3*df_out[out_col].std(), \
                             df_out[out_col].mean() + 3*df_out[out_col].std()

            n_affected = mask.sum()

            if out_action == "Cap / Winsorise (clip to fences)":
                df_out[out_col] = df_out[out_col].clip(lower=lo_f, upper=hi_f)
                msg = f"Capped {n_affected} outliers in '{out_col}' to [{lo_f:.3f}, {hi_f:.3f}]"
            elif out_action == "Fill with Mean":
                fill_val = df_out[out_col].mean()
                df_out.loc[mask, out_col] = fill_val
                msg = f"Filled {n_affected} outliers in '{out_col}' with mean ({fill_val:.4f})"
            elif out_action == "Fill with Median":
                fill_val = df_out[out_col].median()
                df_out.loc[mask, out_col] = fill_val
                msg = f"Filled {n_affected} outliers in '{out_col}' with median ({fill_val:.4f})"
            elif out_action == "Fill with Mode":
                fill_val = df_out[out_col].mode().iloc[0]
                df_out.loc[mask, out_col] = fill_val
                msg = f"Filled {n_affected} outliers in '{out_col}' with mode ({fill_val:.4f})"
            elif out_action == "Remove rows with outliers":
                df_out = df_out[~mask].reset_index(drop=True)
                msg = f"Removed {n_affected} rows with outliers in '{out_col}'"

            st.session_state.df_clean = df_out
            st.session_state.clean_log.append(f"✅ {msg}")
            st.success(msg)
            st.rerun()

    st.divider()

    # ── SECTION D: CLEANING LOG ────────────────────────────────
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

    # ── SECTION E: ML-Ready export ────────────────────────────
    st.divider()
    st.markdown('<div class="section-header">🤖 Encoding Preview & ML-Ready Export</div>', unsafe_allow_html=True)

    enc_method = st.selectbox("Encoding Method (preview only)",
        ["Label Encoding","One-Hot Encoding","Ordinal Encoding","Frequency Encoding"])
    enc_col = st.selectbox("Column to Preview",
        cat_cols if cat_cols else ["(no categorical columns)"])

    if cat_cols and enc_col in cat_cols:
        from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
        s_enc = df_work[enc_col].fillna("Missing")
        if enc_method == "Label Encoding":
            le = LabelEncoder()
            preview = pd.DataFrame({enc_col: s_enc, "Encoded": le.fit_transform(s_enc)})
        elif enc_method == "One-Hot Encoding":
            preview = pd.get_dummies(s_enc, prefix=enc_col).astype(int)
            preview.insert(0, enc_col, s_enc.values)
        elif enc_method == "Frequency Encoding":
            freq_map = s_enc.value_counts(normalize=True)
            preview  = pd.DataFrame({enc_col: s_enc, "Freq_Encoded": s_enc.map(freq_map).round(4)})
        else:
            cats = sorted(s_enc.unique())
            oe = OrdinalEncoder(categories=[cats])
            preview = pd.DataFrame({enc_col: s_enc,
                "Ordinal": oe.fit_transform(s_enc.values.reshape(-1,1)).flatten()})
        st.dataframe(preview.head(30), use_container_width=True)

    if st.button("💾 Save ML-Ready as .pkl"):
        miss_pct = df_work.isna().mean() * 100
        payload = {
            "dataframe": df_work, "num_cols": num_cols,
            "cat_cols": cat_cols, "date_cols": date_cols, "shape": df_work.shape,
            "missing_pct": miss_pct.to_dict(),
        }
        pkl_buf = io.BytesIO()
        pickle.dump(payload, pkl_buf)
        pkl_buf.seek(0)
        st.download_button("⬇️ Download ml_ready.pkl", pkl_buf,
            file_name="ml_ready.pkl", mime="application/octet-stream")


# ══════════════════════════════════════════════
#  TAB 8 — EXPORT
# ══════════════════════════════════════════════
with tabs[7]:
    st.markdown('<div class="section-header">Export Data</div>', unsafe_allow_html=True)

    export_df_choice = st.radio("Export Which Dataset?",
        ["Original (filtered/renamed)", "Cleaned (after Data Cleaning tab)"])

    exp_df = df_work if export_df_choice.startswith("Original") else st.session_state.df_clean[keep]

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
#  TAB 9 — RECOMMENDATIONS  (ALL NEW)
# ══════════════════════════════════════════════
with tabs[8]:
    st.markdown("### 💡 Automated Data Recommendations")
    st.caption("Scans your dataset and gives step-by-step, actionable guidance with exact Python code.")

    # ── Compute all stats for the report ──────────────────────
    all_num_stats = {col: numeric_full_stats(df_work[col]) for col in num_cols}
    miss_pct_rec  = (df_work.isna().mean() * 100).round(2)
    dupes_n_rec   = df_work.duplicated().sum()

    severity_counts = {"critical": 0, "warning": 0, "ok": 0}

    def render_card(level, title, body):
        cls_map = {"critical":"rec-card-danger","warning":"rec-card-warn","ok":"rec-card-ok","info":"rec-card"}
        icon_map = {"critical":"🔴","warning":"🟡","ok":"🟢","info":"🔵"}
        severity_counts[level if level in severity_counts else "ok"] += 1
        st.markdown(
            f'<div class="{cls_map.get(level,"rec-card")}">'
            f'<b>{icon_map.get(level,"ℹ️")} {title}</b><br>{body}</div>',
            unsafe_allow_html=True
        )

    # ── 1. DUPLICATE ROWS ──────────────────────────────────────
    st.markdown('<div class="section-header">1️⃣ Duplicate Rows</div>', unsafe_allow_html=True)
    if dupes_n_rec == 0:
        render_card("ok", "No duplicate rows", "Dataset is free of duplicate rows. ✅")
    else:
        pct = round(dupes_n_rec/len(df_work)*100, 2)
        render_card("critical" if pct > 5 else "warning",
            f"{dupes_n_rec:,} duplicate rows detected ({pct}%)",
            f"""Duplicates can skew statistics and inflate model performance artificially.<br>
<b>Fix:</b> Go to the 🧹 Data Cleaning tab → <i>Duplicate Removal</i> section, or run:<br>
<code>df.drop_duplicates(inplace=True)</code>""")

    # ── 2. MISSING VALUES ──────────────────────────────────────
    st.markdown('<div class="section-header">2️⃣ Missing Values</div>', unsafe_allow_html=True)
    high_miss_cols = miss_pct_rec[miss_pct_rec > 30].index.tolist()
    mod_miss_cols  = miss_pct_rec[(miss_pct_rec > 0) & (miss_pct_rec <= 30)].index.tolist()

    if not high_miss_cols and not mod_miss_cols:
        render_card("ok", "No missing values", "All columns are complete. ✅")
    for col in high_miss_cols:
        render_card("critical", f"'{col}' has {miss_pct_rec[col]:.1f}% missing — consider dropping",
            f"""More than 30% missing — this column may not be reliable.<br>
<b>Options:</b><br>
• Drop the column: <code>df.drop('{col}', axis=1, inplace=True)</code><br>
• Drop rows: <code>df.dropna(subset=['{col}'], inplace=True)</code><br>
• Impute if justified: <code>df['{col}'].fillna(df['{col}'].median(), inplace=True)</code>""")
    for col in mod_miss_cols:
        is_num = col in num_cols
        if is_num:
            st_vals = all_num_stats.get(col, {})
            skew = abs(st_vals.get("skewness", 0))
            rec_fill = "Median" if skew > 1 else "Mean"
            code_val = f"df['{col}'].median()" if skew > 1 else f"df['{col}'].mean()"
            render_card("warning", f"'{col}' has {miss_pct_rec[col]:.1f}% missing",
                f"""Numeric column with missing data. Skewness = {st_vals.get('skewness','?')} → recommend <b>{rec_fill}</b> imputation.<br>
<code>df['{col}'].fillna({code_val}, inplace=True)</code>""")
        else:
            render_card("warning", f"'{col}' has {miss_pct_rec[col]:.1f}% missing (categorical)",
                f"""<b>Options:</b><br>
• Fill with mode: <code>df['{col}'].fillna(df['{col}'].mode()[0], inplace=True)</code><br>
• Fill with 'Unknown': <code>df['{col}'].fillna('Unknown', inplace=True)</code>""")

    # ── 3. SKEWNESS ────────────────────────────────────────────
    st.markdown('<div class="section-header">3️⃣ Skewness & Normality</div>', unsafe_allow_html=True)
    skew_found = False
    for col in num_cols:
        st_v = all_num_stats.get(col, {})
        skew_v = st_v.get("skewness", 0)
        if abs(skew_v) > 1:
            skew_found = True
            direction  = "right (+)" if skew_v > 0 else "left (–)"
            transform  = "np.log1p(df['{col}'])" if skew_v > 0 else f"np.log1p(df['{col}'].max() - df['{col}'])"
            render_card("warning", f"'{col}' is highly skewed (skewness = {skew_v})",
                f"""Skewed {direction}. Affects regression models and mean-based statistics.<br>
<b>Step-by-step fix:</b><br>
1. Check histogram: look for long tail<br>
2. Try log transform: <code>df['{col}_log'] = np.log1p(df['{col}'])</code><br>
3. Or use Yeo-Johnson: <code>from sklearn.preprocessing import PowerTransformer<br>
   pt = PowerTransformer(method='yeo-johnson')<br>
   df['{col}_yj'] = pt.fit_transform(df[['{col}']])</code><br>
4. Re-check: <code>df['{col}_log'].skew()</code> — aim for |skew| &lt; 0.5<br>
5. Also check the Q-Q plot in the Charts tab for normality visual.""")
        elif not st_v.get("is_normal", True) and abs(skew_v) > 0.5:
            skew_found = True
            render_card("info", f"'{col}' is moderately skewed (skewness = {skew_v})",
                f"""Slight skew detected. May still work for many models but worth monitoring.<br>
<b>Check normality:</b> <code>from scipy.stats import shapiro; shapiro(df['{col}'].dropna())</code>""")

    if not skew_found:
        render_card("ok", "All numeric columns have acceptable skewness", "No highly skewed columns detected. ✅")

    # ── 4. OUTLIERS ────────────────────────────────────────────
    st.markdown('<div class="section-header">4️⃣ Outlier Detection</div>', unsafe_allow_html=True)
    outlier_found = False
    for col in num_cols:
        st_v = all_num_stats.get(col, {})
        n_iqr = st_v.get("outliers_iqr", 0)
        n_z   = st_v.get("outliers_z", 0)
        if n_iqr > 0:
            outlier_found = True
            pct_out = round(n_iqr/df_work[col].dropna().shape[0]*100, 2)
            lo_f    = st_v.get("lower_fence", "?")
            hi_f    = st_v.get("upper_fence", "?")
            render_card("warning" if pct_out < 5 else "critical",
                f"'{col}' has {n_iqr} IQR outlier(s) ({pct_out}%)",
                f"""Outlier fences: [{lo_f}, {hi_f}]  |  Z-score outliers: {n_z}<br>
<b>Options (pick one):</b><br>
• Cap (Winsorise): <code>df['{col}'] = df['{col}'].clip(lower={lo_f}, upper={hi_f})</code><br>
• Fill with median: <code>mask = (df['{col}'] &lt; {lo_f}) | (df['{col}'] &gt; {hi_f})<br>
  df.loc[mask, '{col}'] = df['{col}'].median()</code><br>
• Fill with mean: replace <code>.median()</code> with <code>.mean()</code><br>
• Remove rows: <code>df = df[~mask].reset_index(drop=True)</code><br>
➡️ Use <b>Cap</b> if you want to keep all rows. Use <b>Remove</b> only if they are data entry errors.""")
    if not outlier_found:
        render_card("ok", "No significant outliers detected (IQR method)", "All numeric columns look clean. ✅")

    # ── 5. HIGH CARDINALITY ────────────────────────────────────
    st.markdown('<div class="section-header">5️⃣ Categorical Cardinality</div>', unsafe_allow_html=True)
    card_found = False
    for col in cat_cols:
        n_uniq = df_work[col].nunique()
        if n_uniq > 50:
            card_found = True
            render_card("warning", f"'{col}' has high cardinality ({n_uniq} unique values)",
                f"""One-Hot Encoding will create {n_uniq} new columns — too many for most models.<br>
<b>Better approaches:</b><br>
• <b>Frequency Encoding:</b> <code>freq = df['{col}'].value_counts(normalize=True)<br>
  df['{col}_freq'] = df['{col}'].map(freq)</code><br>
• <b>Target Encoding</b> (with a target column): use category_encoders library<br>
• <b>Group rare values:</b> <code>top = df['{col}'].value_counts().head(20).index<br>
  df['{col}'] = df['{col}'].where(df['{col}'].isin(top), other='Other')</code>""")
        elif n_uniq == 1:
            render_card("critical", f"'{col}' has only 1 unique value — constant column",
                f"This column carries no information. Drop it: <code>df.drop('{col}', axis=1, inplace=True)</code>")
    if not card_found:
        render_card("ok", "Categorical cardinality is acceptable", "All categorical columns have manageable unique counts. ✅")

    # ── 6. ZERO / LOW VARIANCE ─────────────────────────────────
    st.markdown('<div class="section-header">6️⃣ Low / Zero Variance Columns</div>', unsafe_allow_html=True)
    low_var_found = False
    for col in num_cols:
        st_v = all_num_stats.get(col, {})
        if st_v.get("std", 1) < 1e-6:
            low_var_found = True
            render_card("critical", f"'{col}' has zero variance (constant)",
                f"This column adds no predictive power. <code>df.drop('{col}', axis=1, inplace=True)</code>")
        elif st_v.get("cv_pct") is not None and abs(st_v["cv_pct"]) < 1:
            low_var_found = True
            render_card("info", f"'{col}' has very low variance (CV = {st_v['cv_pct']}%)",
                f"Consider whether this column is meaningful before including in models.")
    if not low_var_found:
        render_card("ok", "All numeric columns have adequate variance", "✅")

    # ── 7. CLASS IMBALANCE HINT ────────────────────────────────
    st.markdown('<div class="section-header">7️⃣ Class Distribution (Categorical)</div>', unsafe_allow_html=True)
    imb_found = False
    for col in cat_cols:
        vc  = df_work[col].value_counts(normalize=True) * 100
        dom = vc.iloc[0]
        if dom > 80:
            imb_found = True
            render_card("warning" if dom < 90 else "critical",
                f"'{col}' is dominated by '{vc.index[0]}' ({dom:.1f}%)",
                f"""If this is a <b>target/label column</b>, your dataset is class-imbalanced.<br>
<b>Fixes:</b><br>
• Oversample minority: <code>from imblearn.over_sampling import SMOTE</code><br>
• Undersample majority: <code>from imblearn.under_sampling import RandomUnderSampler</code><br>
• Use class_weight='balanced' in sklearn models<br>
• Evaluate with F1, AUC-ROC instead of accuracy""")
    if not imb_found:
        render_card("ok", "No severe class imbalance detected", "Categorical columns look balanced. ✅")

    # ── 8. CORRELATION WARNINGS ───────────────────────────────
    if len(num_cols) >= 2:
        st.markdown('<div class="section-header">8️⃣ Multicollinearity (High Correlation)</div>', unsafe_allow_html=True)
        corr_mat = df_work[num_cols].corr().abs()
        high_corr_pairs = []
        for i in range(len(corr_mat.columns)):
            for j in range(i+1, len(corr_mat.columns)):
                r = corr_mat.iloc[i, j]
                if r > 0.85:
                    high_corr_pairs.append((corr_mat.columns[i], corr_mat.columns[j], round(r, 3)))
        if high_corr_pairs:
            for c1n, c2n, r_val in high_corr_pairs:
                render_card("warning", f"High correlation: '{c1n}' ↔ '{c2n}' (r = {r_val})",
                    f"""Multicollinearity can destabilise linear models and inflate feature importance.<br>
<b>Options:</b><br>
• Drop one: <code>df.drop('{c2n}', axis=1, inplace=True)</code><br>
• Use PCA to combine correlated features<br>
• Use regularisation (Ridge/Lasso) which handles multicollinearity""")
        else:
            render_card("ok", "No highly correlated pairs (r > 0.85)", "No multicollinearity risk detected. ✅")

    # ── 9. FULL CHECKLIST ──────────────────────────────────────
    st.markdown('<div class="section-header">✅ Complete EDA Checklist</div>', unsafe_allow_html=True)
    checklist_items = [
        ("Check shape & dtypes",                     True),
        ("Handle missing values",                    not (miss_pct_rec > 0).any()),
        ("Remove duplicate rows",                    dupes_n_rec == 0),
        ("Detect & treat outliers (IQR / Z-score)",  all(all_num_stats.get(c,{}).get("outliers_iqr",0)==0 for c in num_cols)),
        ("Check skewness → transform if needed",     all(abs(all_num_stats.get(c,{}).get("skewness",0))<1 for c in num_cols)),
        ("Verify normality (Q-Q plot, Shapiro-Wilk)", all(all_num_stats.get(c,{}).get("is_normal",True) for c in num_cols)),
        ("Encode categorical columns",               False),
        ("Check correlation / multicollinearity",    True),
        ("Scale/normalise features",                 False),
        ("Check class balance (if classification)",  True),
        ("Export cleaned dataset (.pkl / .csv)",     False),
    ]
    for item, done in checklist_items:
        icon = "✅" if done else "⬜"
        st.markdown(f"{icon} {item}")

    # ── 10. SUMMARY SCORECARD ─────────────────────────────────
    st.markdown('<div class="section-header">📊 Data Quality Scorecard</div>', unsafe_allow_html=True)
    total_issues = severity_counts["critical"] + severity_counts["warning"]
    score = max(0, 100 - severity_counts["critical"]*15 - severity_counts["warning"]*5)
    grade = "A" if score >= 90 else "B" if score >= 75 else "C" if score >= 60 else "D"
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Quality Score", f"{score}/100")
    col2.metric("Grade",          grade)
    col3.metric("Critical Issues",severity_counts["critical"])
    col4.metric("Warnings",       severity_counts["warning"])

    if score == 100:
        st.success("🎉 Your dataset is in excellent shape and ready for analysis or modelling!")
    elif score >= 75:
        st.info(f"📈 Good dataset quality. Address {total_issues} issue(s) before modelling.")
    else:
        st.warning(f"⚠️ Several issues need attention before this data is modelling-ready.")
