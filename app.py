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
    apply_transforms, apply_encoding, encoding_preview,
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
#  CORRECTION TYPE INFO  (new — mirrors ENCODING_INFO style)
# ─────────────────────────────────────────────
CORRECTION_TYPE_INFO = {
    "Fill with Mean": {
        "full_name": "Mean Imputation",
        "what": "Replaces missing values with the arithmetic average of the non-null values in that column.",
        "when": "Data is numeric and approximately normally distributed with few outliers.",
        "why": "Preserves the column mean, minimises distortion for symmetric distributions.",
        "avoid": "Skewed data or columns with heavy outliers — the mean will be pulled toward extremes.",
        "pros": [
            "Simple and fast — one line of code",
            "Works well for normally distributed data",
            "Doesn't alter the column mean",
        ],
        "cons": [
            "Reduces variance and correlation with other variables",
            "Poor choice for skewed or bimodal distributions",
            "Inflates confidence in imputed values",
        ],
        "best_for": ["Symmetric numeric columns", "Low missing % (< 5%)"],
    },
    "Fill with Median": {
        "full_name": "Median Imputation",
        "what": "Replaces missing values with the middle value of the sorted non-null data.",
        "when": "Data is numeric but skewed, or contains outliers.",
        "why": "The median is robust to outliers — it stays at the distribution centre even when extremes exist.",
        "avoid": "Normally distributed data where mean imputation would be equally valid and more intuitive.",
        "pros": [
            "Robust to outliers and skewed distributions",
            "Doesn't inflate extreme values",
            "Good default for numeric columns",
        ],
        "cons": [
            "Still reduces variance",
            "Not suitable for categorical data",
            "May not reflect true central tendency for multimodal data",
        ],
        "best_for": ["Right/left-skewed numeric columns", "Columns with outliers", "Income, salary, price columns"],
    },
    "Fill with Mode": {
        "full_name": "Mode Imputation",
        "what": "Replaces missing values with the most frequently occurring value in the column.",
        "when": "Column is categorical, boolean, or a low-cardinality integer.",
        "why": "The mode is the most 'typical' value — a sensible default for discrete data.",
        "avoid": "High-cardinality columns or numeric columns with a uniform distribution — the mode may not be representative.",
        "pros": [
            "Works for both numeric and categorical columns",
            "Easy to interpret and explain",
            "Preserves the most common category",
        ],
        "cons": [
            "If two modes exist, only one is chosen — can be arbitrary",
            "Over-represents the dominant category",
            "Poor for continuous numeric data",
        ],
        "best_for": ["Categorical / text columns", "Boolean (yes/no) columns", "Low-cardinality integers"],
    },
    "Fill with Constant": {
        "full_name": "Constant / Custom Value Imputation",
        "what": "Fills all missing values with a user-specified constant (e.g. 0, -1, 'Unknown', 'Missing').",
        "when": "Missing values have a known or meaningful interpretation (e.g. 0 means 'none', 'Unknown' flags absence).",
        "why": "Preserves the information that a value was missing, rather than hiding it behind a statistic.",
        "avoid": "When the constant could be confused with a real value (e.g. filling age with 0).",
        "pros": [
            "Full control — you decide the fill value",
            "Makes missingness explicit (e.g. 'Unknown' category)",
            "Useful when absence itself is informative",
        ],
        "cons": [
            "Wrong constant choice can introduce bias",
            "Downstream models may misinterpret the constant as a real value",
        ],
        "best_for": ["When missingness is not at random", "Categorical flags", "When you have domain knowledge about the fill value"],
    },
    "Fill with Forward Fill (ffill)": {
        "full_name": "Forward Fill (Last Observation Carried Forward)",
        "what": "Propagates the last known non-null value forward to fill subsequent missing values.",
        "when": "Data has a natural ordering (time series, sequences) and consecutive rows are related.",
        "why": "In ordered data, the last observed value is the best estimate for the immediately following missing one.",
        "avoid": "Unordered tabular data where rows are independent — forward fill will introduce spurious patterns.",
        "pros": [
            "Ideal for time-series and sensor data",
            "Preserves trend continuity",
            "No statistical assumptions needed",
        ],
        "cons": [
            "Meaningless for unordered / cross-sectional data",
            "Can propagate stale values over long gaps",
            "First row remains NaN if it is the missing one",
        ],
        "best_for": ["Time series", "Sequential measurements", "Log / event data"],
    },
    "Fill with Backward Fill (bfill)": {
        "full_name": "Backward Fill (Next Observation Carried Backward)",
        "what": "Fills missing values using the next available non-null value that appears later in the sequence.",
        "when": "Ordered data where the next known value is a better estimate than the previous one.",
        "why": "Sometimes future context is more reliable — e.g. filling a missing month's sales with next month's figure.",
        "avoid": "Unordered datasets, or when using future values would constitute data leakage in a predictive model.",
        "pros": [
            "Useful when future values are known at inference time",
            "Complements ffill for full gap coverage",
        ],
        "cons": [
            "Risk of data leakage in ML pipelines",
            "Last row remains NaN if it is missing",
            "Only meaningful for ordered data",
        ],
        "best_for": ["Filling start-of-period gaps in time series", "When used after ffill to cover remaining NaNs"],
    },
    "Drop rows with missing": {
        "full_name": "Complete Case Analysis (Listwise Deletion)",
        "what": "Removes any row that contains at least one missing value in the selected column.",
        "when": "The percentage of missing values is very small (< 1–2%) and rows are missing completely at random (MCAR).",
        "why": "Dropping a few rows has negligible impact on analysis when missingness is truly random.",
        "avoid": "When missing % is high (> 5%) or when data is not MCAR — dropping may introduce selection bias.",
        "pros": [
            "Simplest approach — no imputed values to worry about",
            "Keeps data genuine with no synthetic values",
        ],
        "cons": [
            "Loses real data — can reduce statistical power",
            "Introduces bias if missingness is not random (MAR / MNAR)",
            "Dangerous with high missing rates",
        ],
        "best_for": ["< 2% missing values", "MCAR (Missing Completely At Random) scenario", "Small datasets where every column is critical"],
    },
}


# ─────────────────────────────────────────────
#  SKEWNESS CLASSIFICATION REFERENCE
# ─────────────────────────────────────────────
SKEWNESS_CLASSIFICATION = {
    "Highly Right Skewed":      {"range": "skewness > 1",        "color": "#dc2626", "badge": "badge-danger"},
    "Moderately Right Skewed":  {"range": "0.5 < skewness ≤ 1",  "color": "#f97316", "badge": "badge-warn"},
    "Approximately Normal":     {"range": "|skewness| ≤ 0.5",    "color": "#16a34a", "badge": "badge-ok"},
    "Moderately Left Skewed":   {"range": "−1 ≤ skewness < −0.5","color": "#f97316", "badge": "badge-warn"},
    "Highly Left Skewed":       {"range": "skewness < −1",       "color": "#dc2626", "badge": "badge-danger"},
}

# ─────────────────────────────────────────────
#  SKEWNESS TRANSFORMATION INFO
# ─────────────────────────────────────────────
SKEWNESS_TRANSFORM_INFO = {
    "Log (log1p)": {
        "full_name": "Logarithmic Transformation — log(x + 1)",
        "what": "Applies the natural logarithm after adding 1 to handle zeros: log(x + 1).",
        "when": "Column is right-skewed (positively skewed) with all values ≥ 0.",
        "why": "Compresses the long right tail, pulling extreme high values closer to the bulk of the data. Adding 1 avoids log(0) = −∞.",
        "avoid": "Columns with negative values (requires shift first). Left-skewed data — log makes it worse.",
        "pros": [
            "Most intuitive and widely used de-skewing tool",
            "Handles zero values safely with log1p",
            "Interpreted as proportional / percentage change",
        ],
        "cons": [
            "Only fixes right skew — not left skew",
            "Transformed scale is harder to interpret directly",
            "Negative values require an extra shift step",
        ],
        "best_for": ["Income, salary, price columns", "Counts and frequencies", "Any right-skewed column with values ≥ 0"],
    },
    "Square Root": {
        "full_name": "Square Root Transformation — √x",
        "what": "Applies √x to each value. A milder transformation than log.",
        "when": "Right-skewed data, especially count data or data following a Poisson distribution.",
        "why": "Compresses the right tail less aggressively than log — good when the skew is moderate rather than extreme.",
        "avoid": "Negative values (produces NaN). Extreme skew where a stronger transformation (log, Box-Cox) is needed.",
        "pros": [
            "Milder than log — preserves more of the original scale",
            "Good for count / Poisson data",
            "Simple and fast",
        ],
        "cons": [
            "Doesn't fully fix extreme right skew",
            "Cannot handle negative values",
            "Less effective than log for heavy-tailed distributions",
        ],
        "best_for": ["Moderate right skew", "Count data (views, clicks, events)", "Poisson-distributed columns"],
    },
    "Box-Cox": {
        "full_name": "Box-Cox Power Transformation",
        "what": "Finds the optimal power parameter λ that makes the distribution most normal. When λ=0 it is log; λ=0.5 is sqrt; λ=1 is no change.",
        "when": "Data is right-skewed and strictly positive (all values > 0).",
        "why": "Automatically chooses the best power transformation — more flexible than manually picking log or sqrt.",
        "avoid": "Data containing zeros or negative values (requires a positive shift first). When interpretability matters — the λ-transformed scale is hard to explain.",
        "pros": [
            "Optimal — finds the best λ for normality automatically",
            "More flexible than log or sqrt",
            "Supported by scipy and sklearn",
        ],
        "cons": [
            "Requires all values > 0 (fails on zeros without shift)",
            "Transformed values are hard to interpret",
            "λ must be re-estimated on new data (leakage risk in ML)",
        ],
        "best_for": ["Strictly positive right-skewed columns", "When log/sqrt don't fully fix normality", "Preprocessing for linear/regression models"],
    },
    "Yeo-Johnson": {
        "full_name": "Yeo-Johnson Power Transformation",
        "what": "Generalisation of Box-Cox that works on both positive AND negative values by splitting the transformation at zero.",
        "when": "Column can contain zeros or negative values; you need a power transformation more flexible than log.",
        "why": "Unlike Box-Cox, it doesn't require strictly positive values — making it a safer default for general numeric columns.",
        "avoid": "When interpretability is critical — the transformed scale is as opaque as Box-Cox.",
        "pros": [
            "Works with zeros AND negative values",
            "Optimal λ chosen automatically",
            "Sklearn's PowerTransformer uses Yeo-Johnson by default",
        ],
        "cons": [
            "Transformed scale is uninterpretable",
            "λ must be fitted — can't apply to new data without refitting",
            "Slightly more complex than log/sqrt",
        ],
        "best_for": ["Columns with negative values", "General-purpose normality transformation", "sklearn preprocessing pipelines"],
    },
    "Standard Scaling (Z-score)": {
        "full_name": "Standardisation — Z-score Normalisation",
        "what": "Transforms each value to z = (x − mean) / std, producing a distribution with mean 0 and std 1.",
        "when": "Features need to be on the same scale for distance-based models (SVM, KNN, PCA, regularised regression).",
        "why": "Does NOT change the shape of the distribution — skewness is unchanged. Only shifts and rescales.",
        "avoid": "When you want to actually fix skewness — use log/sqrt/Box-Cox first, then scale. Also avoid when the model is tree-based (trees are scale-invariant).",
        "pros": [
            "Required by many ML algorithms (SVM, KNN, PCA, logistic regression)",
            "Preserves outlier information (unlike MinMax)",
            "Interpretable as 'standard deviations from the mean'",
        ],
        "cons": [
            "Does NOT fix skewness",
            "Sensitive to outliers (outliers pull mean and std)",
            "Useless for tree-based models",
        ],
        "best_for": ["Pre-ML feature scaling", "PCA / dimensionality reduction", "Regularised regression (Ridge, Lasso)"],
    },
    "MinMax Scaling [0,1]": {
        "full_name": "Min-Max Normalisation",
        "what": "Rescales all values to the [0, 1] range using x_scaled = (x − min) / (max − min).",
        "when": "Model requires features in a bounded range (neural networks, image pixel values).",
        "why": "Ensures all features contribute equally when the algorithm is sensitive to absolute magnitude.",
        "avoid": "When outliers are present — one extreme value compresses all others into a tiny range. Does NOT fix skewness.",
        "pros": [
            "Bounded output — always in [0,1]",
            "Useful for neural networks and gradient-based models",
            "Preserves exact zero values",
        ],
        "cons": [
            "Extremely sensitive to outliers",
            "Does NOT change distribution shape or fix skewness",
            "Range changes if new data has different min/max",
        ],
        "best_for": ["Neural networks / deep learning", "Image data normalisation", "Algorithms that need features in [0,1]"],
    },
    "Robust Scaling": {
        "full_name": "Robust Scaler (IQR-based)",
        "what": "Scales using median and IQR: x_scaled = (x − median) / IQR. Outliers have minimal influence.",
        "when": "Data has significant outliers that would distort Z-score or MinMax scaling.",
        "why": "Uses robust statistics (median, IQR) instead of mean and std, so extreme values don't dominate the scaling.",
        "avoid": "When the algorithm requires a strict [0,1] range — robust scaler doesn't guarantee bounded output.",
        "pros": [
            "Robust to outliers — uses median & IQR",
            "Better than Z-score for skewed data with outliers",
            "Centred around median (not mean)",
        ],
        "cons": [
            "Output range is unbounded",
            "Still doesn't fix underlying skewness",
            "Less commonly understood than Z-score",
        ],
        "best_for": ["Numeric columns with known outliers", "When Z-score gives poor results due to outliers", "Pre-processing before outlier-sensitive algorithms"],
    },
}


# ─────────────────────────────────────────────
#  CORRELATION METHOD INFO
# ─────────────────────────────────────────────
CORRELATION_METHOD_INFO = {
    "pearson": {
        "full_name": "Pearson Product-Moment Correlation",
        "what": "Measures the linear relationship between two continuous variables. Ranges from −1 (perfect negative) to +1 (perfect positive); 0 means no linear relationship.",
        "when": "Both variables are continuous, approximately normally distributed, and their relationship is expected to be linear.",
        "why": "The most commonly used correlation; sensitive to the actual magnitude of values, making it the most powerful test when assumptions are met.",
        "avoid": "Ordinal data, non-linear relationships, or data with significant outliers — Pearson is heavily influenced by extreme values.",
        "pros": [
            "Most statistically powerful when normality holds",
            "Produces a coefficient that is easy to interpret",
            "Widely used and understood",
        ],
        "cons": [
            "Assumes linearity — misses curved relationships",
            "Sensitive to outliers",
            "Assumes approximate normality for significance testing",
        ],
        "best_for": ["Continuous numeric columns", "Height vs weight", "Temperature vs energy use", "Linear relationships"],
    },
    "spearman": {
        "full_name": "Spearman Rank Correlation (ρ)",
        "what": "Measures the monotonic relationship between two variables by ranking the data first, then computing Pearson correlation on the ranks.",
        "when": "Data is ordinal, non-normally distributed, or the relationship is monotonic but not necessarily linear.",
        "why": "By converting to ranks, outliers have minimal influence and any monotonic relationship (not just linear) is captured.",
        "avoid": "When you specifically need to measure the strength of a linear relationship — Spearman will detect any monotonic relationship, which could be misleading.",
        "pros": [
            "Robust to outliers",
            "Works with ordinal data",
            "Captures any monotonic relationship",
            "No normality assumption",
        ],
        "cons": [
            "Less powerful than Pearson when normality truly holds",
            "Detects monotonic but not strictly linear patterns",
            "Ties in data can reduce accuracy",
        ],
        "best_for": ["Skewed distributions", "Ordinal survey data (Likert scales)", "Data with outliers", "Non-linear but monotonic relationships"],
    },
    "kendall": {
        "full_name": "Kendall Rank Correlation (τ)",
        "what": "Measures ordinal association by comparing all pairs of observations and counting concordant vs discordant pairs.",
        "when": "Small samples, heavily tied data, or when you need a more conservative (robust) rank-based measure than Spearman.",
        "why": "Kendall's τ has a more direct probabilistic interpretation (probability of concordance minus probability of discordance) and is more robust with ties.",
        "avoid": "Large datasets — Kendall is O(n²) and slow. When Spearman is sufficient and speed matters.",
        "pros": [
            "Most robust to ties among rank methods",
            "Direct probability interpretation",
            "Better for small samples",
            "More conservative — less likely to overstate correlation",
        ],
        "cons": [
            "Computationally expensive — O(n²)",
            "Typically produces smaller τ values than Spearman ρ for the same data",
            "Less familiar to most analysts",
        ],
        "best_for": ["Small samples (n < 30)", "Heavy ties in data", "Ordinal variables with few levels", "When a conservative correlation estimate is preferred"],
    },
}


# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None
if "clean_log" not in st.session_state:
    st.session_state.clean_log = []
if "df_clean_history" not in st.session_state:
    st.session_state.df_clean_history = []  # list of (df_snapshot, log_snapshot)

# ── Tab metadata ──────────────────────────────────────────────────────────────
TAB_NAMES = [
    "🗂 Overview",
    "🔢 Numerical",
    "🏷️ Categorical",
    "📐 Correlation",
    "📊 Charts",
    "🔄 Transform",
    "🧹 Data Cleaning",
    "➕ Add Data",
    "💾 Export",
    "💡 Recommendations",
]


def _save_snapshot():
    """Push the current df_clean + log onto the undo stack (max 10 snapshots)."""
    if st.session_state.df_clean is not None:
        st.session_state.df_clean_history.append(
            (st.session_state.df_clean.copy(), list(st.session_state.clean_log))
        )
        # Keep last 10 snapshots to avoid excessive memory usage
        if len(st.session_state.df_clean_history) > 10:
            st.session_state.df_clean_history.pop(0)


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
    """Render a styled info card for an outlier, encoding, or correction method."""
    pros  = "".join(f'<li>✅ {p}</li>' for p in info_dict.get("pros", []))
    cons  = "".join(f'<li>❌ {c}</li>' for c in info_dict.get("cons", []))
    best  = "".join(f'<span class="tag">{b}</span>' for b in info_dict.get("best_for", []))
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


def _compute_quality_score(df, num_cols_list):
    """
    Recomputes the data quality score locally — identical formula to DataPrep Pro's
    calculate_data_quality_score(), so the gauge always matches the breakdown bars.
    Returns (score_0_to_100, factor_scores_dict, factor_max_dict).
    """
    miss_pct = df.isnull().sum().sum() / df.size * 100 if df.size else 0
    dup_pct  = df.duplicated().sum() / len(df) * 100 if len(df) else 0

    # Average IQR outlier % across numeric columns
    avg_out_pct = 0.0
    if num_cols_list:
        out_pcts = []
        for col in num_cols_list:
            s = df[col].dropna()
            if len(s) < 4:
                continue
            Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
            IQR = Q3 - Q1
            lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            pct = ((s < lo) | (s > hi)).mean() * 100
            out_pcts.append(pct)
        avg_out_pct = float(np.mean(out_pcts)) if out_pcts else 0.0

    # Invalid / domain values (non-negative keyword heuristic matching DataPrep Pro)
    NON_NEG_KW = ["age", "salary", "income", "revenue", "debt", "experience",
                   "weight", "height", "price", "cost", "quantity", "amount",
                   "months", "years", "tumor", "size", "count", "duration",
                   "population", "rate", "pct", "percent", "score", "grade",
                   "rank", "distance", "area", "volume", "length", "width"]
    SIGNED_KW  = ["profit", "loss", "temperature", "balance", "change", "growth",
                   "return", "difference", "delta", "variance", "gain", "net",
                   "flow", "deviation", "residual"]

    inv_count = 0
    for col in num_cols_list:
        cl = col.lower().replace(" ", "_")
        if any(kw in cl for kw in SIGNED_KW):
            continue
        if any(kw in cl for kw in NON_NEG_KW):
            inv_count += int((df[col] < 0).sum())
    inv_pct = inv_count / len(df) * 100 if len(df) else 0

    factor_scores = {
        "Missing Values (30 pts)":    max(0.0, 30.0 - miss_pct * 0.6),
        "Duplicates (20 pts)":        max(0.0, 20.0 - dup_pct  * 0.4),
        "Outliers (20 pts)":          max(0.0, 20.0 - avg_out_pct * 0.4),
        "Invalid / Domain (15 pts)":  max(0.0, 15.0 - inv_pct  * 0.3),
        "Consistency (15 pts)":       15.0,
    }
    factor_max = {
        "Missing Values (30 pts)":   30,
        "Duplicates (20 pts)":       20,
        "Outliers (20 pts)":         20,
        "Invalid / Domain (15 pts)": 15,
        "Consistency (15 pts)":      15,
    }
    total = round(min(100.0, sum(factor_scores.values())), 1)
    return total, factor_scores, factor_max


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
        pct     = round(score_val / max_val * 100, 1) if max_val else 0
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

        # ── Live quality snapshot in sidebar (uses local recompute) ───────────
        st.markdown("---")
        qs_sidebar, _, _ = _compute_quality_score(df_work, num_cols)
        rec_sidebar = build_recommendations(df_work, num_cols, cat_cols)
        dup_sidebar = rec_sidebar["duplicates"]["count"]
        qc = "#16a34a" if qs_sidebar >= 80 else "#d97706" if qs_sidebar >= 60 else "#dc2626"
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
                <span style="font-family:'JetBrains Mono',monospace;font-size:0.82rem;font-weight:700;color:#dc2626;">{dup_sidebar}</span>
            </div>
            <div style="margin-top:12px;">
                <div style="font-size:0.7rem;color:#64748b;margin-bottom:5px;">Quality Score</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:1.5rem;font-weight:700;color:{qc};">{qs_sidebar}<span style="font-size:0.85rem;font-weight:400;color:#94a3b8;">/100</span></div>
                <div style="background:#f1f5f9;border-radius:6px;height:6px;margin-top:5px;">
                    <div style="width:{qs_sidebar}%;height:100%;background:{qc};border-radius:6px;"></div>
                </div>
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

tabs = st.tabs(TAB_NAMES)


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
        metric_card(dupes, "Duplicate Rows", "#dc2626" if dupes > 0 else "#16a34a")
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
        # ── Method description expander ────────────────────────
        with st.expander("📖 Learn about correlation methods — Pearson vs Spearman vs Kendall", expanded=False):
            corr_method_tabs = st.tabs(list(CORRELATION_METHOD_INFO.keys()))
            for tab_cm, (m_name, m_info) in zip(corr_method_tabs, CORRELATION_METHOD_INFO.items()):
                with tab_cm:
                    render_method_card(m_name, m_info)

        # ── Quick decision helper ──────────────────────────────
        st.markdown("""
        <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;
                    padding:14px 16px;margin:0 0 16px 0;font-size:0.85rem;color:#334155;">
            <b style="color:#16a34a;">🧭 Quick Guide:</b>
            &nbsp;<b>Pearson</b> → continuous + normally distributed + linear relationship &nbsp;|&nbsp;
            <b>Spearman</b> → skewed / ordinal / outliers &nbsp;|&nbsp;
            <b>Kendall</b> → small samples or many tied values
        </div>
        """, unsafe_allow_html=True)

        corr_cols   = st.multiselect("Select Columns", num_cols, default=num_cols)
        corr_method = st.selectbox("Correlation Method", ["pearson", "spearman", "kendall"])
        sig_level   = st.slider("Significance Level α", 0.01, 0.10, 0.05, 0.01)

        # Show active method info badge
        method_info_active = CORRELATION_METHOD_INFO.get(corr_method, {})
        st.markdown(f"""
        <div style="background:#eef2ff;border:1px solid #c7d2fe;border-radius:8px;
                    padding:10px 14px;margin:8px 0 14px 0;font-size:0.83rem;color:#334155;">
            <b style="color:#6366f1;">📌 {corr_method.capitalize()}</b> —
            {method_info_active.get('what','')}<br>
            <span style="color:#64748b;font-size:0.78rem;">
                <b>Best for:</b> {"  ·  ".join(method_info_active.get('best_for', []))}
                &nbsp;|&nbsp;
                <b>Avoid when:</b> {method_info_active.get('avoid', '—')}
            </span>
        </div>
        """, unsafe_allow_html=True)

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
            corr_pairs_df = correlation_pairs(df_work, corr_cols, method=corr_method, sig_level=sig_level)
            st.dataframe(corr_pairs_df, use_container_width=True)

            # ── Interpretation guide ──────────────────────────
            st.markdown("""
            <div style="margin-top:16px;padding:14px 16px;background:#fff;border:1px solid #e2e8f0;
                        border-radius:10px;font-size:0.84rem;color:#334155;">
                <b>📏 Correlation Strength Guide:</b><br>
                <span style="color:#dc2626;font-weight:600;">|r| ≥ 0.90</span> — Very strong &nbsp;·&nbsp;
                <span style="color:#f97316;font-weight:600;">0.70 ≤ |r| &lt; 0.90</span> — Strong &nbsp;·&nbsp;
                <span style="color:#d97706;font-weight:600;">0.50 ≤ |r| &lt; 0.70</span> — Moderate &nbsp;·&nbsp;
                <span style="color:#6366f1;font-weight:600;">0.30 ≤ |r| &lt; 0.50</span> — Weak &nbsp;·&nbsp;
                <span style="color:#16a34a;font-weight:600;">|r| &lt; 0.30</span> — Negligible<br><br>
                ⚠️ <b>Correlation ≠ Causation.</b>
                A high correlation between two variables does not mean one causes the other —
                always check for confounders and use domain knowledge.
            </div>
            """, unsafe_allow_html=True)



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
    st.markdown("### 🔄 Skewness Analysis & Transformations")
    st.caption("Identify distribution shape for every numeric column, then apply the right transformation to fix it.")

    if not num_cols:
        st.warning("No numeric columns to transform.")
    else:
        # ── A: Skewness overview table ─────────────────────────
        st.markdown('<div class="section-header">〰️ Skewness Overview — All Numeric Columns</div>',
                    unsafe_allow_html=True)

        # Reference card for skewness interpretation
        with st.expander("📖 How to read skewness — classification guide", expanded=False):
            cols_sk_ref = st.columns(5)
            for col_ref, (cls_name, cls_info) in zip(cols_sk_ref, SKEWNESS_CLASSIFICATION.items()):
                with col_ref:
                    st.markdown(f"""
                    <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;
                                padding:12px;text-align:center;margin:4px 0;">
                        <div style="font-size:0.75rem;font-weight:700;color:{cls_info['color']};
                                    margin-bottom:6px;">{cls_name}</div>
                        <div style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                                    color:#64748b;background:#f1f5f9;border-radius:6px;
                                    padding:3px 6px;">{cls_info['range']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown("""
            <div style="margin-top:12px;padding:12px 16px;background:#eff6ff;border:1px solid #bfdbfe;
                        border-radius:8px;font-size:0.85rem;color:#334155;">
                <b>Rule of thumb:</b> For ML models that assume normality (linear regression, logistic regression, LDA, Gaussian Naive Bayes),
                aim for <b>|skewness| &lt; 0.5</b> after transformation.
                Tree-based models (Random Forest, XGBoost, LightGBM) are <b>not affected by skewness</b>.
            </div>
            """, unsafe_allow_html=True)

        # Build skewness table
        skew_rows = []
        for col in num_cols:
            s = df_work[col].dropna()
            sk = float(s.skew())
            if sk < -1:        cls = "Highly Left Skewed"
            elif sk < -0.5:    cls = "Moderately Left Skewed"
            elif sk <= 0.5:    cls = "Approximately Normal"
            elif sk <= 1:      cls = "Moderately Right Skewed"
            else:              cls = "Highly Right Skewed"
            rec_t = ("Log (log1p)" if sk > 1 else
                     "Square Root" if sk > 0.5 else
                     "Yeo-Johnson" if sk < -0.5 else
                     "None needed")
            skew_rows.append({
                "Column": col, "Skewness": round(sk, 4),
                "Classification": cls, "Recommended Transform": rec_t,
                "Mean": round(float(s.mean()), 4), "Median": round(float(s.median()), 4),
                "Std": round(float(s.std()), 4),
            })
        skew_df = pd.DataFrame(skew_rows)

        # Colour the Classification column
        def _colour_cls(val):
            colour_map = {
                "Highly Right Skewed":     "background-color:#fef2f2;color:#dc2626;font-weight:600",
                "Highly Left Skewed":      "background-color:#fef2f2;color:#dc2626;font-weight:600",
                "Moderately Right Skewed": "background-color:#fffbeb;color:#d97706;font-weight:600",
                "Moderately Left Skewed":  "background-color:#fffbeb;color:#d97706;font-weight:600",
                "Approximately Normal":    "background-color:#f0fdf4;color:#16a34a;font-weight:600",
            }
            return colour_map.get(val, "")
        
        st.dataframe(
            skew_df.style.map(_colour_cls, subset=["Classification"]).format(
                {"Skewness": "{:.4f}", "Mean": "{:.4f}", "Median": "{:.4f}", "Std": "{:.4f}"}
            ),
            use_container_width=True, height=min(400, 60 + 35 * len(skew_rows)),
        )

        # ── B: Per-column distribution charts with skew annotation ──
        st.markdown('<div class="section-header">📊 Distribution Histograms + KDE</div>',
                    unsafe_allow_html=True)
        st.caption("Red dashed line = Mean · Green dotted line = Median · Title colour = skew severity")

        n_plot_cols = 2
        plot_rows   = [num_cols[i:i+n_plot_cols] for i in range(0, len(num_cols), n_plot_cols)]
        for row_cols in plot_rows:
            fig_cols = st.columns(len(row_cols))
            for fc, col in zip(fig_cols, row_cols):
                with fc:
                    s    = df_work[col].dropna()
                    sk   = float(s.skew())
                    info = next((r for r in skew_rows if r["Column"] == col), {})
                    cls  = info.get("Classification", "")
                    t_color = SKEWNESS_CLASSIFICATION.get(cls, {}).get("color", "#6366f1")

                    try:
                        from scipy.stats import gaussian_kde as _gkde
                        fig_h = go.Figure()
                        fig_h.add_trace(go.Histogram(
                            x=s, nbinsx=30, name="Distribution",
                            marker_color="rgba(99,102,241,0.45)",
                            marker_line=dict(color="rgba(99,102,241,0.8)", width=0.5),
                        ))
                        kde_fn = _gkde(s)
                        xs     = np.linspace(float(s.min()), float(s.max()), 250)
                        kde_y  = kde_fn(xs) * len(s) * (float(s.max()) - float(s.min())) / 30
                        fig_h.add_trace(go.Scatter(
                            x=xs, y=kde_y, mode="lines", name="KDE",
                            line=dict(color=t_color, width=2.5),
                        ))
                        fig_h.add_vline(x=float(s.mean()),   line_dash="dash", line_color="#dc2626", line_width=1.5)
                        fig_h.add_vline(x=float(s.median()), line_dash="dot",  line_color="#16a34a", line_width=1.5)
                        fig_h.update_layout(
                            title=dict(text=f"{col}<br><sup style='color:{t_color}'>{cls} (sk={sk:.3f})</sup>",
                                       font=dict(size=12)),
                            template=template, height=280, showlegend=False,
                            margin=dict(l=10, r=10, t=50, b=30),
                            paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
                        )
                        st.plotly_chart(fig_h, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Chart error for '{col}': {e}")

        # ── C: Transformation descriptions ────────────────────────
        st.markdown('<div class="section-header">📚 Transformation Method Guide</div>',
                    unsafe_allow_html=True)
        with st.expander("📖 Learn about every transformation — which one fixes your skew?", expanded=False):
            transform_tabs = st.tabs(list(SKEWNESS_TRANSFORM_INFO.keys()))
            for tab_t, (t_name, t_info) in zip(transform_tabs, SKEWNESS_TRANSFORM_INFO.items()):
                with tab_t:
                    render_method_card(t_name, t_info)

        # ── D: Apply transformation ────────────────────────────────
        st.markdown('<div class="section-header">⚡ Apply Transformation</div>', unsafe_allow_html=True)

        t_col_sel   = st.selectbox("Select Column to Transform", num_cols, key="t_col_main")
        t_col_skew  = next((r["Skewness"] for r in skew_rows if r["Column"] == t_col_sel), 0)
        t_col_cls   = next((r["Classification"] for r in skew_rows if r["Column"] == t_col_sel), "")
        t_col_rec   = next((r["Recommended Transform"] for r in skew_rows if r["Column"] == t_col_sel), "")

        sk_color = SKEWNESS_CLASSIFICATION.get(t_col_cls, {}).get("color", "#6366f1")
        st.markdown(f"""
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
                    padding:14px 16px;margin:8px 0 16px 0;display:flex;gap:24px;flex-wrap:wrap;">
            <div>
                <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;">Skewness</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:1.3rem;font-weight:700;color:{sk_color};">{t_col_skew:+.4f}</div>
            </div>
            <div>
                <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;">Classification</div>
                <div style="font-size:0.9rem;font-weight:600;color:{sk_color};margin-top:4px;">{t_col_cls}</div>
            </div>
            <div>
                <div style="font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:1px;">Recommended</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.9rem;font-weight:700;
                            color:#6366f1;margin-top:4px;">{t_col_rec}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        all_transforms = list(SKEWNESS_TRANSFORM_INFO.keys())
        default_idx    = all_transforms.index(t_col_rec) if t_col_rec in all_transforms else 0
        transforms_sel = st.multiselect(
            "Transformations to Apply (select one or more to compare)",
            all_transforms,
            default=[all_transforms[default_idx]],
            key="transforms_multisel",
        )

        if t_col_sel and transforms_sel:
            df_transformed, preview, errors = apply_transforms(df_work, t_col_sel, transforms_sel)

            for t_name, err_msg in errors.items():
                st.warning(f"⚠️ Could not apply '{t_name}': {err_msg}")

            if preview:
                st.markdown("**Before vs After — Summary Statistics:**")
                before_stats = {
                    "original": {
                        "mean":   round(float(df_work[t_col_sel].mean()), 4),
                        "median": round(float(df_work[t_col_sel].median()), 4),
                        "std":    round(float(df_work[t_col_sel].std()), 4),
                        "skew":   round(float(df_work[t_col_sel].skew()), 4),
                    }
                }
                st.dataframe(
                    pd.DataFrame({**before_stats, **preview}).T
                    .rename_axis("Transform").reset_index(),
                    use_container_width=True,
                )

                # Side-by-side before/after charts
                chart_cols = st.columns(len(transforms_sel) + 1)
                orig_series = df_work[t_col_sel].dropna()
                with chart_cols[0]:
                    fig_b = go.Figure(go.Histogram(
                        x=orig_series, nbinsx=30,
                        marker_color="rgba(99,102,241,0.5)", name="Original",
                    ))
                    fig_b.update_layout(title=f"Original<br><sup>sk={orig_series.skew():.3f}</sup>",
                                        template=template, height=240, showlegend=False,
                                        margin=dict(l=5,r=5,t=45,b=20),
                                        paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc")
                    st.plotly_chart(fig_b, use_container_width=True)

                for i, t_name in enumerate(transforms_sel, 1):
                    if i < len(chart_cols):
                        col_name = f"{t_col_sel}_{t_name.split()[0].lower()}"
                        if col_name not in df_transformed.columns:
                            # Try to find by partial match
                            matches = [c for c in df_transformed.columns if c.startswith(t_col_sel + "_") and c != t_col_sel]
                            col_name = matches[i-1] if i-1 < len(matches) else None
                        if col_name and col_name in df_transformed.columns:
                            after_s = df_transformed[col_name].dropna()
                            after_sk = round(float(after_s.skew()), 3)
                            after_color = ("#16a34a" if abs(after_sk) <= 0.5
                                           else "#d97706" if abs(after_sk) <= 1
                                           else "#dc2626")
                            with chart_cols[i]:
                                fig_a = go.Figure(go.Histogram(
                                    x=after_s, nbinsx=30,
                                    marker_color=f"rgba(22,163,74,0.5)", name=t_name,
                                ))
                                fig_a.update_layout(
                                    title=f"{t_name}<br><sup style='color:{after_color}'>sk={after_sk}</sup>",
                                    template=template, height=240, showlegend=False,
                                    margin=dict(l=5,r=5,t=45,b=20),
                                    paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
                                )
                                st.plotly_chart(fig_a, use_container_width=True)

                st.success("✅ Transformations previewed above.")

            # ── Apply & Save button ──────────────────────────────
            st.markdown("**Save transformation(s) to the working dataset:**")
            if st.button("💾 Apply & Save Transformation(s)", key="btn_apply_transform"):
                _save_snapshot()
                df_save = st.session_state.df_clean.copy()
                saved_cols = []
                for t_name in transforms_sel:
                    df_save, _, t_errors = apply_transforms(df_save, t_col_sel, [t_name])
                    for t_n, err in t_errors.items():
                        st.warning(f"⚠️ Could not apply '{t_n}': {err}")
                    # figure out the new column name that was added
                    suffix_map = {
                        "Log (log1p)":               f"{t_col_sel}_log1p",
                        "Square Root":               f"{t_col_sel}_sqrt",
                        "Box-Cox":                   f"{t_col_sel}_boxcox",
                        "Yeo-Johnson":               f"{t_col_sel}_yeojohnson",
                        "Standard Scaling (Z-score)":f"{t_col_sel}_zscore",
                        "MinMax Scaling [0,1]":      f"{t_col_sel}_minmax",
                        "Robust Scaling":            f"{t_col_sel}_robust",
                    }
                    new_col = suffix_map.get(t_name)
                    if new_col and new_col in df_save.columns:
                        saved_cols.append(new_col)
                st.session_state.df_clean = df_save
                for sc in saved_cols:
                    st.session_state.clean_log.append(
                        f"✅ Transform applied: '{t_col_sel}' → '{sc}'"
                    )
                if saved_cols:
                    st.success(f"Saved {len(saved_cols)} new column(s): {', '.join(saved_cols)}. "
                               f"Dataset now has {df_save.shape[1]} columns.")
                st.rerun()



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
        _save_snapshot()
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

    # ── NEW: Imputation strategy info expander (mirrors outlier expander style) ──
    with st.expander("📖 Learn about imputation / correction strategies — which one to choose?", expanded=False):
        corr_tabs = st.tabs(list(CORRECTION_TYPE_INFO.keys()))
        for tab_c, (method_name, info) in zip(corr_tabs, CORRECTION_TYPE_INFO.items()):
            with tab_c:
                render_method_card(method_name, info)

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
            _save_snapshot()
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
            _save_snapshot()
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

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("↩️ Undo Last Step", key="btn_undo",
                         disabled=len(st.session_state.df_clean_history) == 0):
                prev_df, prev_log = st.session_state.df_clean_history.pop()
                st.session_state.df_clean = prev_df
                st.session_state.clean_log = prev_log
                st.success("Last operation undone.")
                st.rerun()
        with btn_col2:
            if st.button("🔄 Reset to Original Data", key="btn_reset"):
                st.session_state.df_clean = df_raw.copy()
                st.session_state.clean_log = []
                st.session_state.df_clean_history = []
                st.success("Dataset reset to original.")
                st.rerun()

    st.divider()

    # ── E: ML-Ready / Encoding Preview ────────────────────────
    st.markdown('<div class="section-header">🤖 Encoding & ML-Ready Export</div>',
                unsafe_allow_html=True)

    with st.expander("📖 Learn about encoding methods — which one to choose?", expanded=False):
        enc_tabs = st.tabs(list(ENCODING_INFO.keys()))
        for tab_e, (enc_name, info) in zip(enc_tabs, ENCODING_INFO.items()):
            with tab_e:
                render_method_card(enc_name, info)

    if cat_cols:
        enc_col = st.selectbox("Column to Preview / Encode", cat_cols, key="enc_col_select")

        if enc_col in cat_cols:
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
                "Encoding Method",
                ["Label Encoding", "One-Hot Encoding", "Ordinal Encoding", "Frequency Encoding"],
                index=["Label Encoding","One-Hot Encoding","Ordinal Encoding","Frequency Encoding"].index(
                    enc_rec["recommended"]) if enc_rec["recommended"] in
                    ["Label Encoding","One-Hot Encoding","Ordinal Encoding","Frequency Encoding"] else 0,
                key="enc_method_select",
            )

            # Ordinal order input
            ordinal_order_input = None
            if enc_method == "Ordinal Encoding":
                unique_vals = sorted(df_clean_work[enc_col].dropna().astype(str).unique())
                st.caption(f"Unique values detected: {', '.join(unique_vals)}")
                ordinal_str = st.text_input(
                    "Define ordinal order (comma-separated, lowest → highest)",
                    value=", ".join(unique_vals),
                    key="ordinal_order_input",
                    help="Enter category names separated by commas in ascending order. "
                         "E.g.: Low, Medium, High",
                )
                ordinal_order_input = [v.strip() for v in ordinal_str.split(",") if v.strip()]

            # Live preview
            st.markdown("**Preview (first 30 rows):**")
            preview_enc = encoding_preview(df_work[enc_col], enc_col, enc_method)
            st.dataframe(preview_enc.head(30), use_container_width=True)

            # Apply & Save button
            st.markdown("**Apply encoding to the working dataset:**")
            c_enc1, c_enc2 = st.columns([2, 1])
            with c_enc1:
                if enc_method == "One-Hot Encoding":
                    st.caption(
                        f"ℹ️ One-Hot will **replace** '{enc_col}' with "
                        f"{enc_rec['n_unique']} binary columns."
                    )
                else:
                    st.caption(
                        f"ℹ️ A new encoded column will be **added** alongside '{enc_col}'. "
                        f"Original column is preserved."
                    )
            with c_enc2:
                if st.button("✅ Apply Encoding", key="btn_apply_enc"):
                    _save_snapshot()
                    df_enc, enc_log = apply_encoding(
                        st.session_state.df_clean,
                        enc_col,
                        enc_method,
                        ordinal_order=ordinal_order_input,
                    )
                    st.session_state.df_clean = df_enc
                    st.session_state.clean_log.append(f"✅ {enc_log}")
                    st.success(enc_log)
                    st.rerun()
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
#  TAB 7 — ADD DATA
# ══════════════════════════════════════════════
with tabs[7]:
    st.markdown("### ➕ Add Data to Dataset")
    st.caption("Manually add new rows to your working dataset. Changes are saved to the cleaned dataset and reflected across all tabs.")

    if df_raw is None:
        st.info("Upload a dataset first to add rows.")
    else:
        df_add_work = st.session_state.df_clean[keep].copy()
        num_c_add, cat_c_add, _, _ = detect_col_types(df_add_work)

        st.markdown('<div class="section-header">📋 Current Dataset</div>', unsafe_allow_html=True)
        add_m1, add_m2, add_m3 = st.columns(3)
        with add_m1:
            metric_card(f"{len(df_add_work):,}", "Total Rows")
        with add_m2:
            metric_card(f"{len(df_add_work.columns)}", "Columns")
        with add_m3:
            metric_card(f"{df_add_work.isnull().sum().sum():,}", "Missing Values")

        st.markdown("&nbsp;")

        # ── Number of rows to add ─────────────────────────────
        st.markdown('<div class="section-header">✍️ Enter New Row(s)</div>', unsafe_allow_html=True)
        n_rows_to_add = st.number_input(
            "How many rows do you want to add?", min_value=1, max_value=50, value=1, step=1,
            key="n_rows_add"
        )

        st.info(f"Fill in values for each column. Leave blank to insert a missing value (NaN).")

        # Build a form for each new row
        new_rows_data = []
        for row_i in range(int(n_rows_to_add)):
            if n_rows_to_add > 1:
                st.markdown(f"**Row {row_i + 1}**")
            row_vals = {}
            # Split columns into groups of 3 for a neat grid
            col_groups = [df_add_work.columns.tolist()[i:i+3]
                          for i in range(0, len(df_add_work.columns), 3)]
            for grp in col_groups:
                form_cols = st.columns(len(grp))
                for fc, col in zip(form_cols, grp):
                    with fc:
                        col_dtype = str(df_add_work[col].dtype)
                        is_numeric = col in num_c_add
                        is_cat     = col in cat_c_add
                        placeholder = f"{col}"

                        if is_numeric:
                            # Show numeric input with optional blank
                            raw_val = st.text_input(
                                f"{col} *(numeric)*",
                                value="",
                                placeholder="e.g. 42.5",
                                key=f"add_row{row_i}_{col}",
                            )
                            if raw_val.strip() == "":
                                row_vals[col] = np.nan
                            else:
                                try:
                                    row_vals[col] = float(raw_val)
                                except ValueError:
                                    st.warning(f"'{col}': '{raw_val}' is not numeric — stored as NaN")
                                    row_vals[col] = np.nan
                        elif is_cat:
                            unique_vals = df_add_work[col].dropna().unique().tolist()
                            if len(unique_vals) <= 30:
                                options = ["(blank / NaN)"] + [str(v) for v in sorted(unique_vals)]
                                chosen = st.selectbox(
                                    f"{col} *(categorical)*",
                                    options,
                                    key=f"add_row{row_i}_{col}",
                                )
                                row_vals[col] = np.nan if chosen == "(blank / NaN)" else chosen
                            else:
                                raw_val = st.text_input(
                                    f"{col} *(categorical)*",
                                    value="",
                                    placeholder="Type a value",
                                    key=f"add_row{row_i}_{col}",
                                )
                                row_vals[col] = np.nan if raw_val.strip() == "" else raw_val.strip()
                        else:
                            raw_val = st.text_input(
                                f"{col}",
                                value="",
                                placeholder="Enter value",
                                key=f"add_row{row_i}_{col}",
                            )
                            row_vals[col] = np.nan if raw_val.strip() == "" else raw_val.strip()

            new_rows_data.append(row_vals)
            if n_rows_to_add > 1 and row_i < n_rows_to_add - 1:
                st.markdown("---")

        # ── Preview of new rows ───────────────────────────────
        st.markdown('<div class="section-header">👁️ Preview New Row(s)</div>', unsafe_allow_html=True)
        preview_new = pd.DataFrame(new_rows_data, columns=df_add_work.columns)
        st.dataframe(preview_new, use_container_width=True)

        # ── CSV paste / bulk import ───────────────────────────
        st.markdown('<div class="section-header">📥 Bulk Import via CSV Paste</div>', unsafe_allow_html=True)
        st.caption("Paste CSV text below (with header row matching your dataset columns) to add multiple rows at once.")

        cols_hint = ", ".join(df_add_work.columns.tolist())
        csv_paste = st.text_area(
            "Paste CSV rows here",
            height=120,
            placeholder=f"Header: {cols_hint}\nRow 1: value1, value2, ...\nRow 2: value1, value2, ...",
            key="csv_paste_input",
        )

        bulk_rows_preview = None
        if csv_paste.strip():
            try:
                import io as _io
                bulk_rows_preview = pd.read_csv(_io.StringIO(csv_paste.strip()))
                st.success(f"✅ Parsed {len(bulk_rows_preview)} row(s) from pasted CSV")
                st.dataframe(bulk_rows_preview, use_container_width=True)
            except Exception as e:
                st.error(f"Could not parse CSV: {e}")
                bulk_rows_preview = None

        # ── Commit buttons ────────────────────────────────────
        st.markdown("&nbsp;")
        btn_col_a, btn_col_b, btn_col_c = st.columns([2, 2, 1])

        with btn_col_a:
            if st.button("✅ Add Manually Entered Row(s)", key="btn_add_manual_rows",
                         use_container_width=True):
                _save_snapshot()
                df_combined = pd.concat(
                    [st.session_state.df_clean, preview_new],
                    ignore_index=True,
                )
                # Coerce numeric columns back
                for col in num_c_add:
                    if col in df_combined.columns:
                        df_combined[col] = pd.to_numeric(df_combined[col], errors="coerce")
                st.session_state.df_clean = df_combined
                st.session_state.clean_log.append(
                    f"➕ Added {len(preview_new)} row(s) manually. "
                    f"Dataset now has {len(df_combined):,} rows."
                )
                st.success(
                    f"✅ {len(preview_new)} row(s) added! Dataset now has {len(df_combined):,} rows."
                )
                st.rerun()

        with btn_col_b:
            if bulk_rows_preview is not None:
                if st.button("📥 Add Bulk CSV Row(s)", key="btn_add_bulk_rows",
                             use_container_width=True):
                    _save_snapshot()
                    df_combined = pd.concat(
                        [st.session_state.df_clean, bulk_rows_preview],
                        ignore_index=True,
                    )
                    for col in num_c_add:
                        if col in df_combined.columns:
                            df_combined[col] = pd.to_numeric(df_combined[col], errors="coerce")
                    st.session_state.df_clean = df_combined
                    st.session_state.clean_log.append(
                        f"➕ Bulk-imported {len(bulk_rows_preview)} row(s) via CSV paste. "
                        f"Dataset now has {len(df_combined):,} rows."
                    )
                    st.success(
                        f"✅ {len(bulk_rows_preview)} row(s) added! Dataset now has {len(df_combined):,} rows."
                    )
                    st.rerun()

        with btn_col_c:
            if st.button("↩️ Undo", key="btn_undo_add",
                         disabled=len(st.session_state.df_clean_history) == 0,
                         use_container_width=True):
                prev_df, prev_log = st.session_state.df_clean_history.pop()
                st.session_state.df_clean = prev_df
                st.session_state.clean_log = prev_log
                st.success("Last add operation undone.")
                st.rerun()

        # ── Updated dataset preview ───────────────────────────
        st.markdown('<div class="section-header">📊 Updated Dataset (last 20 rows)</div>',
                    unsafe_allow_html=True)
        st.dataframe(st.session_state.df_clean[keep].tail(20), use_container_width=True)



# ══════════════════════════════════════════════
#  TAB 8 — EXPORT  (was TAB 8, now shifted)
# ══════════════════════════════════════════════
with tabs[8]:
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
#  TAB 9 — RECOMMENDATIONS  (fully rebuilt)
# ══════════════════════════════════════════════
with tabs[9]:
    st.markdown("### 💡 Automated Data Recommendations")
    st.caption("Scans your dataset and gives step-by-step, actionable guidance with exact Python code.")

    rec = build_recommendations(df_work, num_cols, cat_cols)

    # ── FIXED: recompute score locally so gauge matches breakdown bars ─────────
    qs_total, factor_scores, factor_max = _compute_quality_score(df_work, num_cols)

    # Derive grade & counts from recomputed score (matches DataPrep Pro logic)
    def _grade(s):
        if s >= 90: return "A"
        if s >= 80: return "B"
        if s >= 70: return "C"
        if s >= 60: return "D"
        return "F"

    # Count critical / warning items from the recommendations dict
    _n_crit = (
        (1 if rec["duplicates"]["count"] > 0 and rec["duplicates"]["pct"] > 5 else 0)
        + sum(1 for m in rec.get("missing", [])    if m["pct"] > 30)
        + sum(1 for o in rec.get("outliers", [])   if o["pct"] >= 5)
        + sum(1 for r in rec.get("cardinality", []) if r["n_unique"] == 1)
    )
    _n_warn = (
        (1 if rec["duplicates"]["count"] > 0 and rec["duplicates"]["pct"] <= 5 else 0)
        + sum(1 for m in rec.get("missing",    []) if 0 < m["pct"] <= 30)
        + sum(1 for s_r in rec.get("skewness", []) if True)
        + sum(1 for o in rec.get("outliers",   []) if o["pct"] < 5)
        + sum(1 for r in rec.get("cardinality",[]) if r["n_unique"] > 50)
        + sum(1 for r in rec.get("high_corr",  []))
    )

    # ── Quality Gauge + Scorecard ──────────────────────────────
    st.markdown('<div class="section-header">📊 Data Quality Scorecard</div>', unsafe_allow_html=True)

    g_col, b_col = st.columns([1, 1])

    with g_col:
        st.plotly_chart(quality_gauge(qs_total), use_container_width=True)

    with b_col:
        st.markdown("&nbsp;")
        qc_top = st.columns(4)
        for qw, label, val, col_hex in zip(
            qc_top,
            ["Score", "Grade", "Critical", "Warnings"],
            [f"{qs_total}/100", _grade(qs_total), _n_crit, _n_warn],
            ["#6366f1",
             "#16a34a" if _grade(qs_total) in ("A","B") else "#d97706",
             "#dc2626" if _n_crit else "#16a34a",
             "#d97706" if _n_warn else "#16a34a"],
        ):
            with qw:
                metric_card(val, label, col_hex)

        st.markdown("&nbsp;")
        quality_breakdown_bars(factor_scores, factor_max)

    # ── Overall verdict ────────────────────────────────────────
    total_issues = _n_crit + _n_warn
    if qs_total >= 90:
        st.success("🎉 Excellent dataset quality — ready for analysis or modelling!")
    elif qs_total >= 75:
        st.info(f"📈 Good quality. Address {total_issues} issue(s) before modelling.")
    elif qs_total >= 60:
        st.warning(f"⚠️ Moderate quality. {_n_crit} critical issues need attention.")
    else:
        st.error(f"🚨 Poor data quality. {_n_crit} critical issues must be fixed before modelling.")

    st.markdown("---")

    # 1. Duplicates ────────────────────────────────────────────
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
    if not rec.get("missing"):
        render_rec_card("ok", "No missing values", "All columns are complete. ✅")
    for m in rec.get("missing", []):
        col   = m["col"]
        pct_m = m["pct"]
        if pct_m > 30:
            render_rec_card("critical", f"'{col}' has {pct_m:.1f}% missing — consider dropping",
                f"More than 30% missing — this column may not be reliable.<br>"
                f"<b>Options:</b><br>"
                f"• Drop column: <code>df.drop('{col}', axis=1, inplace=True)</code><br>"
                f"• Impute: <code>df['{col}'].fillna(df['{col}'].median(), inplace=True)</code>")
        elif m.get("is_numeric"):
            skew_v   = m.get("skewness")
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

    # 3. Skewness ──────────────────────────────────────────────
    st.markdown('<div class="section-header">3️⃣ Skewness & Normality</div>', unsafe_allow_html=True)
    if not rec.get("skewness"):
        render_rec_card("ok", "All numeric columns have acceptable skewness",
                        "No highly skewed columns detected. ✅")
    for s_r in rec.get("skewness", []):
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

    # 4. Outliers ──────────────────────────────────────────────
    st.markdown('<div class="section-header">4️⃣ Outlier Detection</div>', unsafe_allow_html=True)
    if not rec.get("outliers"):
        render_rec_card("ok", "No significant outliers detected (IQR method)",
                        "All numeric columns look clean. ✅")
    for o in rec.get("outliers", []):
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

    # 5. Cardinality ───────────────────────────────────────────
    st.markdown('<div class="section-header">5️⃣ Categorical Cardinality</div>', unsafe_allow_html=True)
    card_high  = [r for r in rec.get("cardinality", []) if r["n_unique"] > 50]
    card_const = [r for r in rec.get("cardinality", []) if r["n_unique"] == 1]
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

    # 6. Low Variance ──────────────────────────────────────────
    st.markdown('<div class="section-header">6️⃣ Low / Zero Variance Columns</div>', unsafe_allow_html=True)
    if not rec.get("low_variance"):
        render_rec_card("ok", "All numeric columns have adequate variance", "✅")
    for r in rec.get("low_variance", []):
        col = r["col"]
        if r["std"] < 1e-6:
            render_rec_card("critical", f"'{col}' has zero variance (constant)",
                f"Drop it: <code>df.drop('{col}', axis=1, inplace=True)</code>")
        else:
            render_rec_card("info", f"'{col}' has very low variance (CV = {r['cv_pct']}%)",
                "Consider whether this column is meaningful before including in models.")

    # 7. Class Imbalance ───────────────────────────────────────
    st.markdown('<div class="section-header">7️⃣ Class Distribution (Categorical)</div>', unsafe_allow_html=True)
    if not rec.get("imbalance"):
        render_rec_card("ok", "No severe class imbalance detected",
                        "Categorical columns look balanced. ✅")
    for r in rec.get("imbalance", []):
        col = r["col"]; dom_val = r["dominant_val"]; pct_i = r["dominant_pct"]
        render_rec_card(
            "critical" if pct_i >= 90 else "warning",
            f"'{col}' dominated by '{dom_val}' ({pct_i:.1f}%)",
            f"If this is a target/label column, the dataset is class-imbalanced.<br>"
            f"<b>Fixes:</b><br>"
            f"• Oversample: <code>from imblearn.over_sampling import SMOTE</code><br>"
            f"• Use <code>class_weight='balanced'</code> in sklearn models<br>"
            f"• Evaluate with F1, AUC-ROC instead of accuracy")

    # 8. Multicollinearity ─────────────────────────────────────
    if len(num_cols) >= 2:
        st.markdown('<div class="section-header">8️⃣ Multicollinearity (High Correlation)</div>',
                    unsafe_allow_html=True)
        if not rec.get("high_corr"):
            render_rec_card("ok", "No highly correlated pairs (r > 0.85)",
                            "No multicollinearity risk detected. ✅")
        for r in rec.get("high_corr", []):
            c1n = r["col_a"]; c2n = r["col_b"]; r_val = r["r"]
            render_rec_card("warning", f"High correlation: '{c1n}' ↔ '{c2n}' (r = {r_val})",
                f"Multicollinearity can destabilise linear models.<br>"
                f"<b>Options:</b><br>"
                f"• Drop one: <code>df.drop('{c2n}', axis=1, inplace=True)</code><br>"
                f"• Use PCA to combine correlated features<br>"
                f"• Use regularisation (Ridge/Lasso)")

    # 9. EDA Checklist ─────────────────────────────────────────
    st.markdown('<div class="section-header">✅ Complete EDA Checklist</div>', unsafe_allow_html=True)
    for item, done in rec.get("checklist", []):
        badge_cls = "badge-ok" if done else "badge-warn"
        icon      = "✅" if done else "⬜"
        st.markdown(
            f'{icon} {item} &nbsp; <span class="badge {badge_cls}">{"Done" if done else "Pending"}</span>',
            unsafe_allow_html=True,
        )
