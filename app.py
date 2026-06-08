# ============================================================
#  app.py — Descriptive Analytics Workbench
#  Run: streamlit run app.py
# ============================================================

import io
import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import streamlit as st
from fpdf import FPDF
import xlsxwriter

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
    code { font-family: 'DM Mono', monospace; }
</style>
""", unsafe_allow_html=True)


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

def fig_to_bytes(fig, fmt="png"):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight", dpi=150)
    buf.seek(0)
    return buf

def df_to_excel_bytes(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False)
    buf.seek(0)
    return buf


# ─────────────────────────────────────────────
#  SIDEBAR — FILE UPLOAD & COLUMN CONTROL
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📂 Dataset")
    upload_mode = st.radio("Mode", ["Single File", "Multiple Files (Merge)"], horizontal=True)

    df_raw = None
    sheet_names = None

    if upload_mode == "Single File":
        uploaded = st.file_uploader("Upload CSV / Excel", type=["csv", "xlsx", "xls"])
        if uploaded:
            result = load_file(uploaded)
            if isinstance(result, tuple):          # Excel
                xl, sheet_names = result
                sheet = st.selectbox("Select Sheet", sheet_names)
                df_raw = xl.parse(sheet)
            else:
                df_raw = result
    else:
        files = st.file_uploader("Upload Multiple CSVs/Excels", type=["csv", "xlsx", "xls"], accept_multiple_files=True)
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

    # ── Column Controls ──────────────────────────────
    if df_raw is not None:
        st.markdown("---")
        st.markdown("### ⚙️ Column Controls")

        all_cols = df_raw.columns.tolist()
        exclude  = st.multiselect("Exclude Columns", all_cols)
        keep     = [c for c in all_cols if c not in exclude]
        df_work  = df_raw[keep].copy()

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
                    f"'{c}' ({cur})", ["(keep)", "numeric", "string", "category", "datetime"],
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
            "Custom Filter (pandas query syntax)",
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
        palette  = st.selectbox("Color Palette", ["viridis", "plasma", "Set2", "tab10", "coolwarm", "Blues", "RdYlGn"])
        dark_mode = st.toggle("Dark Mode", value=False)
        chart_h  = st.slider("Chart Height (px)", 300, 900, 500, 50)
        chart_w  = st.slider("Chart Width (px)",  500, 1400, 900, 50)
        font_sz  = st.slider("Font Size", 8, 20, 12)
        show_grid = st.toggle("Show Grid", value=True)

        bg_color = "#0f172a" if dark_mode else "#ffffff"
        txt_color = "#f1f5f9" if dark_mode else "#1e293b"

        template = "plotly_dark" if dark_mode else "plotly_white"


# ─────────────────────────────────────────────
#  MAIN AREA
# ─────────────────────────────────────────────

st.title("📊 Descriptive Analytics Workbench")
st.caption("Upload your dataset → choose analyses → explore, export, and stay ML-ready.")

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
    "💾 Export",
    "🧹 Data Cleaning",
    "💡 Recommendations",
])


# ══════════════════════════════════════════════
#  TAB 1 — OVERVIEW
# ══════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-header">Dataset Snapshot</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows",    f"{df_work.shape[0]:,}")
    c2.metric("Columns", f"{df_work.shape[1]:,}")
    c3.metric("Numeric", len(num_cols))
    c4.metric("Categorical", len(cat_cols))

    st.dataframe(df_work.head(50), use_container_width=True)

    st.markdown('<div class="section-header">Data Types</div>', unsafe_allow_html=True)
    dtype_df = pd.DataFrame({
        "Column": df_work.columns,
        "Dtype" : df_work.dtypes.astype(str).values,
        "Non-Null Count": df_work.notna().sum().values,
        "Null Count"    : df_work.isna().sum().values,
        "Null %"        : (df_work.isna().mean() * 100).round(2).values,
        "Unique Values" : df_work.nunique().values,
    })
    st.dataframe(dtype_df, use_container_width=True)

    st.markdown('<div class="section-header">Missing Values</div>', unsafe_allow_html=True)
    miss = df_work.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if miss.empty:
        st.success("✅ No missing values found!")
    else:
        fig = px.bar(
            x=miss.index, y=miss.values,
            labels={"x": "Column", "y": "Missing Count"},
            color=miss.values, color_continuous_scale=palette,
            template=template, height=chart_h//1.5
        )
        fig.update_layout(font_size=font_sz, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Duplicate Analysis</div>', unsafe_allow_html=True)
    dupes = df_work.duplicated().sum()
    st.metric("Duplicate Rows", dupes, delta=f"{dupes/len(df_work)*100:.1f}% of data" if dupes else "Clean ✅")
    if dupes:
        st.dataframe(df_work[df_work.duplicated()], use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 2 — NUMERICAL
# ══════════════════════════════════════════════
with tabs[1]:
    if not num_cols:
        st.warning("No numeric columns detected.")
    else:
        sel_num = st.multiselect("Select Numeric Columns", num_cols, default=num_cols[:min(8, len(num_cols))])

        analyses = st.multiselect("Select Analyses", [
            "Descriptive Statistics", "Percentile Table", "Skewness & Kurtosis",
            "Normality Test (Shapiro-Wilk)", "Outlier Detection (IQR)",
            "Outlier Detection (Z-Score)", "Variance Analysis"
        ], default=["Descriptive Statistics"])

        if sel_num:
            df_num = df_work[sel_num]

            if "Descriptive Statistics" in analyses:
                st.markdown('<div class="section-header">Descriptive Statistics</div>', unsafe_allow_html=True)
                desc = df_num.describe().T
                desc["cv%"]     = (desc["std"] / desc["mean"] * 100).round(2)
                desc["range"]   = desc["max"] - desc["min"]
                desc["iqr"]     = df_num.quantile(0.75) - df_num.quantile(0.25)
                desc["mad"]     = df_num.apply(lambda x: (x - x.mean()).abs().mean())
                st.dataframe(desc.style.format("{:.4f}"), use_container_width=True)

            if "Percentile Table" in analyses:
                st.markdown('<div class="section-header">Percentile Table</div>', unsafe_allow_html=True)
                pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
                pct_df = df_num.quantile([p/100 for p in pcts]).T
                pct_df.columns = [f"P{p}" for p in pcts]
                st.dataframe(pct_df.style.format("{:.4f}"), use_container_width=True)

            if "Skewness & Kurtosis" in analyses:
                st.markdown('<div class="section-header">Skewness & Kurtosis</div>', unsafe_allow_html=True)
                sk_df = pd.DataFrame({
                    "Skewness" : df_num.skew().round(4),
                    "Kurtosis" : df_num.kurtosis().round(4),
                    "Skew Interpretation": df_num.skew().apply(
                        lambda x: "Highly Negative" if x < -1 else "Moderate Negative" if x < -0.5
                        else "Approx. Normal" if abs(x) <= 0.5 else "Moderate Positive" if x < 1 else "Highly Positive"
                    )
                })
                st.dataframe(sk_df, use_container_width=True)

            if "Normality Test (Shapiro-Wilk)" in analyses:
                st.markdown('<div class="section-header">Normality Test (Shapiro-Wilk)</div>', unsafe_allow_html=True)
                norm_rows = []
                for col in sel_num:
                    sample = df_num[col].dropna()
                    if len(sample) > 5000:
                        sample = sample.sample(5000, random_state=42)
                    stat, p = stats.shapiro(sample)
                    norm_rows.append({
                        "Column": col, "Statistic": round(stat, 4), "p-value": round(p, 6),
                        "Normal?": "✅ Yes (p > 0.05)" if p > 0.05 else "❌ No (p ≤ 0.05)"
                    })
                st.dataframe(pd.DataFrame(norm_rows), use_container_width=True)

            if "Outlier Detection (IQR)" in analyses:
                st.markdown('<div class="section-header">Outlier Detection — IQR Method</div>', unsafe_allow_html=True)
                iqr_rows = []
                for col in sel_num:
                    s  = df_num[col].dropna()
                    Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
                    IQR = Q3 - Q1
                    lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
                    n_out = ((s < lo) | (s > hi)).sum()
                    iqr_rows.append({
                        "Column": col, "Q1": round(Q1,4), "Q3": round(Q3,4),
                        "IQR": round(IQR,4), "Lower Fence": round(lo,4),
                        "Upper Fence": round(hi,4), "Outlier Count": n_out,
                        "Outlier %": round(n_out/len(s)*100, 2)
                    })
                st.dataframe(pd.DataFrame(iqr_rows), use_container_width=True)

            if "Outlier Detection (Z-Score)" in analyses:
                st.markdown('<div class="section-header">Outlier Detection — Z-Score (|z| > 3)</div>', unsafe_allow_html=True)
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
                    "Variance"  : df_num.var().round(4),
                    "Std Dev"   : df_num.std().round(4),
                    "CV %"      : (df_num.std() / df_num.mean() * 100).round(2)
                })
                st.dataframe(var_df, use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 3 — CATEGORICAL
# ══════════════════════════════════════════════
with tabs[2]:
    if not cat_cols:
        st.warning("No categorical columns detected.")
    else:
        sel_cat = st.multiselect("Select Categorical Columns", cat_cols, default=cat_cols[:min(5, len(cat_cols))])
        cat_analyses = st.multiselect("Select Analyses", [
            "Frequency Table", "Cardinality", "Dominant Category",
            "Rare Categories (<1%)", "Entropy Analysis"
        ], default=["Frequency Table", "Cardinality"])

        for col in sel_cat:
            st.markdown(f'<div class="section-header">📌 {col}</div>', unsafe_allow_html=True)
            s = df_work[col].dropna()
            vc = s.value_counts()
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
                    st.warning(f"{len(rare)} rare categories (< 1%):")
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
        corr_method = st.selectbox("Method", ["pearson", "spearman", "kendall"])
        sig_level   = st.slider("Significance Level α", 0.01, 0.10, 0.05, 0.01)

        if len(corr_cols) >= 2:
            corr_df = df_work[corr_cols].corr(method=corr_method)

            st.markdown('<div class="section-header">Correlation Matrix (Table)</div>', unsafe_allow_html=True)
            st.dataframe(corr_df.style.background_gradient(cmap=palette, axis=None).format("{:.3f}"),
                         use_container_width=True)

            st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
            fig_c = px.imshow(
                corr_df, text_auto=".2f", color_continuous_scale=palette,
                template=template, height=chart_h, width=chart_w,
                title=f"{corr_method.capitalize()} Correlation Heatmap"
            )
            fig_c.update_layout(font_size=font_sz)
            st.plotly_chart(fig_c, use_container_width=True)

            st.markdown('<div class="section-header">Top Correlations (Ranked)</div>', unsafe_allow_html=True)
            pairs = []
            for i in range(len(corr_df.columns)):
                for j in range(i+1, len(corr_df.columns)):
                    c1n, c2n = corr_df.columns[i], corr_df.columns[j]
                    r = corr_df.loc[c1n, c2n]
                    n = df_work[[c1n, c2n]].dropna().shape[0]
                    t_stat = r * np.sqrt(n-2) / np.sqrt(1-r**2+1e-10)
                    p_val  = 2 * stats.t.sf(abs(t_stat), df=n-2)
                    pairs.append({
                        "Col A": c1n, "Col B": c2n,
                        "Correlation": round(r, 4),
                        "p-value": round(p_val, 6),
                        "Significant": "✅" if p_val < sig_level else "❌",
                        "Strength": "Strong" if abs(r) > 0.7 else "Moderate" if abs(r) > 0.4 else "Weak"
                    })
            pairs_df = pd.DataFrame(pairs).sort_values("Correlation", key=abs, ascending=False)
            st.dataframe(pairs_df, use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 5 — CHARTS
# ══════════════════════════════════════════════
with tabs[4]:
    chart_type = st.selectbox("Chart Type", [
        "Histogram", "KDE / Density", "Box Plot", "Violin Plot",
        "Scatter Plot", "Bar Chart", "Pie / Donut Chart",
        "Line Chart", "Area Chart", "Pair Plot",
        "ECDF Plot", "QQ Plot", "Heatmap (Correlation)"
    ])

    col_x = st.selectbox("X / Primary Column", df_work.columns.tolist())
    col_y = st.selectbox("Y Column (if needed)", ["None"] + df_work.columns.tolist())
    col_color = st.selectbox("Color By (optional)", ["None"] + cat_cols)

    chart_title    = st.text_input("Chart Title",    value=chart_type)
    x_label        = st.text_input("X-Axis Label",   value=col_x)
    y_label        = st.text_input("Y-Axis Label",   value=col_y if col_y != "None" else "")
    legend_visible = st.toggle("Show Legend", value=True)

    color_arg  = None if col_color == "None" else col_color
    y_arg      = None if col_y    == "None" else col_y
    fig_chart  = None

    try:
        if chart_type == "Histogram":
            fig_chart = px.histogram(df_work, x=col_x, color=color_arg,
                color_discrete_sequence=px.colors.sequential.__dict__.get(palette, px.colors.qualitative.Safe),
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
            ecdf = np.arange(1, len(sorted_vals)+1) / len(sorted_vals)
            fig_chart = px.line(x=sorted_vals, y=ecdf, labels={"x": x_label, "y": "ECDF"},
                template=template, height=chart_h, title=chart_title)

        elif chart_type == "QQ Plot":
            vals = df_work[col_x].dropna().values
            (osm, osr), (slope, intercept, r) = stats.probplot(vals, dist="norm")
            fig_chart = go.Figure()
            fig_chart.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name="Data"))
            fig_chart.add_trace(go.Scatter(x=osm, y=slope*np.array(osm)+intercept,
                mode="lines", name="Normal Line", line=dict(color="red", dash="dash")))
            fig_chart.update_layout(template=template, height=chart_h, title=chart_title,
                xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles")

        elif chart_type == "Pair Plot":
            sel_pp = st.multiselect("Choose Columns for Pair Plot", num_cols, default=num_cols[:min(5, len(num_cols))])
            if sel_pp:
                fig_chart = px.scatter_matrix(df_work, dimensions=sel_pp, color=color_arg,
                    template=template, height=chart_h, title=chart_title)

        elif chart_type == "Heatmap (Correlation)":
            c_df = df_work[num_cols].corr()
            fig_chart = px.imshow(c_df, text_auto=".2f", color_continuous_scale=palette,
                template=template, height=chart_h, title=chart_title)

        if fig_chart:
            fig_chart.update_layout(
                font_size=font_sz,
                showlegend=legend_visible,
                xaxis_title=x_label,
                yaxis_title=y_label,
                xaxis=dict(showgrid=show_grid),
                yaxis=dict(showgrid=show_grid),
            )
            st.plotly_chart(fig_chart, use_container_width=True)

            # Export chart
            fmt = st.selectbox("Export Chart As", ["png", "svg", "html"])
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
            st.info("Please select valid columns for this chart type.")
    except Exception as e:
        st.error(f"Chart error: {e}")


# ══════════════════════════════════════════════
#  TAB 6 — TRANSFORM
# ══════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-header">Column Transformations</div>', unsafe_allow_html=True)
    st.caption("Transformations apply to a working copy and can be exported.")

    t_col = st.selectbox("Select Column to Transform", num_cols)
    transforms = st.multiselect("Transformations to Apply", [
        "Log (log1p)", "Square Root", "Box-Cox", "Yeo-Johnson",
        "Standard Scaling (Z-score)", "MinMax Scaling [0,1]", "Robust Scaling"
    ])

    df_transformed = df_work.copy()

    if t_col and transforms:
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
        col_data = df_transformed[[t_col]].dropna()

        preview_rows = {}
        for t in transforms:
            try:
                if t == "Log (log1p)":
                    transformed = np.log1p(col_data[t_col])
                    label = f"{t_col}_log1p"
                elif t == "Square Root":
                    transformed = np.sqrt(col_data[t_col].clip(lower=0))
                    label = f"{t_col}_sqrt"
                elif t == "Box-Cox":
                    pt = PowerTransformer(method="box-cox")
                    transformed = pt.fit_transform(col_data[[t_col]]).flatten()
                    label = f"{t_col}_boxcox"
                elif t == "Yeo-Johnson":
                    pt = PowerTransformer(method="yeo-johnson")
                    transformed = pt.fit_transform(col_data[[t_col]]).flatten()
                    label = f"{t_col}_yeojohnson"
                elif t == "Standard Scaling (Z-score)":
                    sc = StandardScaler()
                    transformed = sc.fit_transform(col_data[[t_col]]).flatten()
                    label = f"{t_col}_zscore"
                elif t == "MinMax Scaling [0,1]":
                    sc = MinMaxScaler()
                    transformed = sc.fit_transform(col_data[[t_col]]).flatten()
                    label = f"{t_col}_minmax"
                elif t == "Robust Scaling":
                    sc = RobustScaler()
                    transformed = sc.fit_transform(col_data[[t_col]]).flatten()
                    label = f"{t_col}_robust"

                df_transformed.loc[col_data.index, label] = transformed
                preview_rows[label] = pd.Series(transformed).describe().round(4)
            except Exception as e:
                st.warning(f"Could not apply '{t}': {e}")

        st.dataframe(pd.DataFrame(preview_rows).T, use_container_width=True)
        st.success(f"✅ Transformations applied. Download in Export tab.")


# ══════════════════════════════════════════════
#  TAB 7 — EXPORT
# ══════════════════════════════════════════════
with tabs[6]:
    st.markdown('<div class="section-header">Export Data</div>', unsafe_allow_html=True)

    export_df_choice = st.radio("Export Which Dataset?",
        ["Original (filtered/renamed)", "With Transformations"])
    exp_df = df_work if export_df_choice.startswith("Original") else df_transformed

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
#  TAB 8 — ML READY
# ══════════════════════════════════════════════
with tabs[7]:
    st.markdown('<div class="section-header">ML Readiness Checklist</div>', unsafe_allow_html=True)
    st.caption("This tab keeps your dataset ML-ready without running any model.")

    miss_pct = df_work.isna().mean() * 100
    high_miss = miss_pct[miss_pct > 20].index.tolist()
    low_var   = [c for c in num_cols if df_work[c].std() < 1e-5]
    high_card = [c for c in cat_cols if df_work[c].nunique() > 50]
    dupes_n   = df_work.duplicated().sum()

    issues = []
    if high_miss:  issues.append(f"⚠️ High missing (>20%): {high_miss}")
    if low_var:    issues.append(f"⚠️ Near-zero variance: {low_var}")
    if high_card:  issues.append(f"⚠️ High cardinality categoricals: {high_card}")
    if dupes_n:    issues.append(f"⚠️ {dupes_n} duplicate rows")

    if issues:
        for i in issues:
            st.warning(i)
    else:
        st.success("✅ Dataset looks clean and ML-ready!")

    st.markdown('<div class="section-header">Encoding Preview (no data changed)</div>', unsafe_allow_html=True)
    enc_method = st.selectbox("Encoding Method (preview only)",
        ["Label Encoding", "One-Hot Encoding", "Ordinal Encoding", "Frequency Encoding"])
    enc_col = st.selectbox("Column to Preview", cat_cols if cat_cols else ["(no categorical columns)"])

    if cat_cols and enc_col in cat_cols:
        from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
        s = df_work[enc_col].fillna("Missing")

        if enc_method == "Label Encoding":
            le = LabelEncoder()
            preview = pd.DataFrame({enc_col: s, "Encoded": le.fit_transform(s)})
        elif enc_method == "One-Hot Encoding":
            preview = pd.get_dummies(s, prefix=enc_col).astype(int)
            preview.insert(0, enc_col, s.values)
        elif enc_method == "Frequency Encoding":
            freq_map = s.value_counts(normalize=True)
            preview  = pd.DataFrame({enc_col: s, "Freq_Encoded": s.map(freq_map).round(4)})
        else:  # Ordinal
            cats = sorted(s.unique())
            oe = OrdinalEncoder(categories=[cats])
            preview = pd.DataFrame({enc_col: s, "Ordinal": oe.fit_transform(s.values.reshape(-1,1)).flatten()})

        st.dataframe(preview.head(30), use_container_width=True)

    st.markdown('<div class="section-header">Save ML-Ready Dataset as .pkl</div>', unsafe_allow_html=True)
    if st.button("💾 Save as .pkl (download)"):
        payload = {
            "dataframe"     : df_work,
            "num_cols"      : num_cols,
            "cat_cols"      : cat_cols,
            "date_cols"     : date_cols,
            "shape"         : df_work.shape,
            "missing_pct"   : miss_pct.to_dict(),
        }
        pkl_buf = io.BytesIO()
        pickle.dump(payload, pkl_buf)
        pkl_buf.seek(0)
        st.download_button("⬇️ Download ml_ready.pkl", pkl_buf,
            file_name="ml_ready.pkl", mime="application/octet-stream")
        st.success("pkl includes dataframe + column metadata. Load with `pickle.load()` in your ML notebook.")
