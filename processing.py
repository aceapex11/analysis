# ============================================================
#  processing.py — Backend: all data logic for Analytics Workbench
#  Import this module in app.py; no Streamlit calls live here.
# ============================================================
import io
import pickle
import warnings

import numpy as np
import pandas as pd
import scipy.stats as stats

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  FILE I/O
# ─────────────────────────────────────────────

def load_file(uploaded_file):
    """
    Parse an uploaded CSV or Excel file.
    Returns a DataFrame for CSV, or (ExcelFile, sheet_names) for Excel.
    Tries UTF-8 first, then falls back to latin-1 for CSVs with non-ASCII characters.
    """
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file, encoding="utf-8")
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="latin-1")
    elif name.endswith((".xlsx", ".xls")):
        xl = pd.ExcelFile(uploaded_file)
        return xl, xl.sheet_names
    return None


def merge_files(files, how: str = "vertical") -> pd.DataFrame:
    """
    Merge a list of uploaded files vertically (concat rows) or horizontally (concat cols).
    """
    frames = []
    for f in files:
        result = load_file(f)
        if isinstance(result, tuple):
            xl, sheet_names = result
            frames.append(xl.parse(sheet_names[0]))
        else:
            frames.append(result)
    axis = 0 if how == "vertical" else 1
    return pd.concat(frames, axis=axis, ignore_index=True)


def df_to_excel_bytes(df: pd.DataFrame) -> io.BytesIO:
    """Serialise a DataFrame to an in-memory Excel (.xlsx) buffer."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False)
    buf.seek(0)
    return buf


def df_to_pickle_bytes(payload: dict) -> io.BytesIO:
    """Serialise an arbitrary dict to an in-memory pickle buffer."""
    buf = io.BytesIO()
    pickle.dump(payload, buf)
    buf.seek(0)
    return buf


# ─────────────────────────────────────────────
#  COLUMN / TYPE UTILITIES
# ─────────────────────────────────────────────

def detect_col_types(df: pd.DataFrame):
    """
    Return (num_cols, cat_cols, date_cols, bool_cols) as lists of column names.
    """
    num_cols  = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols  = df.select_dtypes(include=["object", "category"]).columns.tolist()
    date_cols = df.select_dtypes(include=["datetime"]).columns.tolist()
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    return num_cols, cat_cols, date_cols, bool_cols


def apply_renames(df: pd.DataFrame, renames: dict) -> pd.DataFrame:
    """Return a copy of df with columns renamed according to the mapping."""
    return df.rename(columns=renames)


def apply_dtype_changes(df: pd.DataFrame, dtype_map: dict) -> pd.DataFrame:
    """
    Apply dtype conversions.
    dtype_map: {col_name: "numeric" | "string" | "category" | "datetime" | "(keep)"}
    """
    df = df.copy()
    for col, new_dtype in dtype_map.items():
        if new_dtype == "numeric":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif new_dtype == "string":
            df[col] = df[col].astype(str)
        elif new_dtype == "category":
            df[col] = df[col].astype("category")
        elif new_dtype == "datetime":
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def apply_filter(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Apply a pandas query string and return the filtered DataFrame."""
    return df.query(query)


# ─────────────────────────────────────────────
#  STATISTICAL HELPERS
# ─────────────────────────────────────────────

def iqr_fences(s: pd.Series):
    """Return (lower_fence, upper_fence, iqr) for a numeric Series."""
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr, iqr


def numeric_full_stats(s: pd.Series) -> dict:
    """
    Compute a comprehensive stats dict for a numeric Series.
    Includes descriptive stats, skewness, kurtosis, normality test, and outlier counts.
    """
    s = s.dropna()
    if len(s) == 0:
        return {}
    lo, hi, iqr = iqr_fences(s)
    if 3 <= len(s) <= 5000:
        _, p_norm = stats.shapiro(s)
        norm_test = "Shapiro-Wilk"
    else:
        _, p_norm = stats.normaltest(s)
        norm_test = "D'Agostino"
    skew = s.skew()
    kurt = s.kurtosis()
    return {
        "count":    len(s),
        "mean":     s.mean(),
        "median":   s.median(),
        "mode":     s.mode().iloc[0] if not s.mode().empty else None,
        "std":      s.std(),
        "variance": s.var(),
        "min":      s.min(),
        "max":      s.max(),
        "range":    s.max() - s.min(),
        "q1":       s.quantile(0.25),
        "q3":       s.quantile(0.75),
        "iqr":      iqr,
        "p5":       s.quantile(0.05),
        "p95":      s.quantile(0.95),
        "skewness": round(skew, 4),
        "kurtosis": round(kurt, 4),
        "cv_pct":   round(s.std() / s.mean() * 100, 2) if s.mean() != 0 else None,
        "norm_test": norm_test,
        "norm_p":   round(float(p_norm), 5),
        "is_normal": p_norm >= 0.05,
        "outliers_iqr": int(((s < lo) | (s > hi)).sum()),
        "outliers_z":   int((np.abs(stats.zscore(s)) > 3).sum()),
        "lower_fence":  round(lo, 4),
        "upper_fence":  round(hi, 4),
    }


# ─────────────────────────────────────────────
#  OVERVIEW / EDA COMPUTATIONS
# ─────────────────────────────────────────────

def dtype_quality_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-column quality summary DataFrame."""
    return pd.DataFrame({
        "Column":        df.columns,
        "Dtype":         df.dtypes.astype(str).values,
        "Non-Null":      df.notna().sum().values,
        "Null Count":    df.isna().sum().values,
        "Null %":        (df.isna().mean() * 100).round(2).values,
        "Unique Values": df.nunique().values,
    })


def missing_summary(df: pd.DataFrame) -> pd.Series:
    """Return a Series of missing counts for columns that have any."""
    miss = df.isna().sum()
    return miss[miss > 0].sort_values(ascending=False)


def duplicate_rows(df: pd.DataFrame, subset=None) -> int:
    """Return the count of fully-duplicate rows (or subset-duplicate rows)."""
    return int(df.duplicated(subset=subset).sum())


# ─────────────────────────────────────────────
#  NUMERICAL ANALYSIS
# ─────────────────────────────────────────────

def descriptive_stats(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Extended describe() with CV%, range, IQR, and MAD."""
    desc = df[cols].describe().T
    desc["cv%"]   = (desc["std"] / desc["mean"] * 100).round(2)
    desc["range"] = desc["max"] - desc["min"]
    desc["iqr"]   = df[cols].quantile(0.75) - df[cols].quantile(0.25)
    desc["mad"]   = df[cols].apply(lambda x: (x - x.mean()).abs().mean())
    return desc


def percentile_table(df: pd.DataFrame, cols: list,
                     pcts=(1, 5, 10, 25, 50, 75, 90, 95, 99)) -> pd.DataFrame:
    pct_df = df[cols].quantile([p / 100 for p in pcts]).T
    pct_df.columns = [f"P{p}" for p in pcts]
    return pct_df


def skewness_kurtosis(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    def interp(x):
        if x < -1:       return "Highly Negative"
        elif x < -0.5:   return "Moderate Negative"
        elif abs(x) <= 0.5: return "Approx. Normal"
        elif x < 1:      return "Moderate Positive"
        else:            return "Highly Positive"
    return pd.DataFrame({
        "Skewness":              df[cols].skew().round(4),
        "Kurtosis":              df[cols].kurtosis().round(4),
        "Skew Interpretation":   df[cols].skew().apply(interp),
    })


def normality_test(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    rows = []
    for col in cols:
        sample = df[col].dropna()
        if len(sample) > 5000:
            sample = sample.sample(5000, random_state=42)
        stat_v, p = stats.shapiro(sample)
        rows.append({
            "Column":    col,
            "Statistic": round(stat_v, 4),
            "p-value":   round(p, 6),
            "Normal?":   "✅ Yes (p>0.05)" if p > 0.05 else "❌ No (p≤0.05)",
        })
    return pd.DataFrame(rows)


def outlier_iqr_table(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    rows = []
    for col in cols:
        s = df[col].dropna()
        lo, hi, iqr_v = iqr_fences(s)
        n_out = int(((s < lo) | (s > hi)).sum())
        rows.append({
            "Column":       col,
            "Q1":           round(s.quantile(0.25), 4),
            "Q3":           round(s.quantile(0.75), 4),
            "IQR":          round(iqr_v, 4),
            "Lower Fence":  round(lo, 4),
            "Upper Fence":  round(hi, 4),
            "Outlier Count": n_out,
            "Outlier %":    round(n_out / len(s) * 100, 2),
        })
    return pd.DataFrame(rows)


def outlier_zscore_table(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    rows = []
    for col in cols:
        s = df[col].dropna()
        z = np.abs(stats.zscore(s))
        n_out = int((z > 3).sum())
        rows.append({
            "Column":               col,
            "Mean":                 round(s.mean(), 4),
            "Std":                  round(s.std(), 4),
            "Outlier Count (|z|>3)": n_out,
            "Outlier %":            round(n_out / len(s) * 100, 2),
        })
    return pd.DataFrame(rows)


def variance_table(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    return pd.DataFrame({
        "Variance": df[cols].var().round(4),
        "Std Dev":  df[cols].std().round(4),
        "CV %":     (df[cols].std() / df[cols].mean() * 100).round(2),
    })


# ─────────────────────────────────────────────
#  CATEGORICAL ANALYSIS
# ─────────────────────────────────────────────

def frequency_table(s: pd.Series) -> pd.DataFrame:
    vc  = s.value_counts()
    vcp = s.value_counts(normalize=True) * 100
    return pd.DataFrame({
        "Value":     vc.index,
        "Count":     vc.values,
        "Percent %": vcp.values.round(2),
    })


def rare_categories(s: pd.Series, threshold_pct: float = 1.0) -> pd.DataFrame:
    vcp = s.value_counts(normalize=True) * 100
    rare = vcp[vcp < threshold_pct]
    if rare.empty:
        return pd.DataFrame()
    return pd.DataFrame({"Category": rare.index, "Percent %": rare.values.round(3)})


def shannon_entropy(s: pd.Series) -> tuple[float, float]:
    """Return (entropy_bits, max_possible_bits)."""
    from scipy.stats import entropy as sp_entropy
    vc = s.value_counts()
    probs = vc / vc.sum()
    ent = sp_entropy(probs, base=2)
    max_ent = np.log2(s.nunique()) if s.nunique() > 1 else 1
    return float(ent), float(max_ent)


# ─────────────────────────────────────────────
#  CORRELATION
# ─────────────────────────────────────────────

def correlation_pairs(df: pd.DataFrame, cols: list,
                      method: str = "pearson",
                      sig_level: float = 0.05) -> pd.DataFrame:
    """Return a ranked DataFrame of all pairwise correlations with p-values."""
    corr_df = df[cols].corr(method=method)
    pairs = []
    for i in range(len(corr_df.columns)):
        for j in range(i + 1, len(corr_df.columns)):
            c1n, c2n = corr_df.columns[i], corr_df.columns[j]
            r = corr_df.loc[c1n, c2n]
            n = df[[c1n, c2n]].dropna().shape[0]
            t_stat = r * np.sqrt(n - 2) / np.sqrt(1 - r ** 2 + 1e-10)
            p_val  = 2 * stats.t.sf(abs(t_stat), df=n - 2)
            pairs.append({
                "Col A":       c1n,
                "Col B":       c2n,
                "Correlation": round(r, 4),
                "p-value":     round(p_val, 6),
                "Significant": "✅" if p_val < sig_level else "❌",
                "Strength":    "Strong" if abs(r) > 0.7 else "Moderate" if abs(r) > 0.4 else "Weak",
            })
    return pd.DataFrame(pairs).sort_values("Correlation", key=abs, ascending=False)


# ─────────────────────────────────────────────
#  DATA CLEANING OPERATIONS
# ─────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame,
                      subset: list | None = None,
                      keep: str = "first") -> tuple[pd.DataFrame, int]:
    """
    Remove duplicate rows. Returns (cleaned_df, n_removed).
    keep: "first" | "last" | False
    """
    before = len(df)
    df_out = df.drop_duplicates(subset=subset or None, keep=keep)
    df_out = df_out.reset_index(drop=True)
    return df_out, before - len(df_out)


def impute_column(df: pd.DataFrame, col: str, method: str,
                  constant=None) -> pd.DataFrame:
    """
    Impute a single column in-place (on a copy).
    method: "mean" | "median" | "mode" | "constant" | "ffill" | "bfill" | "drop"
    """
    df = df.copy()
    if method == "mean":
        df[col] = df[col].fillna(df[col].mean())
    elif method == "median":
        df[col] = df[col].fillna(df[col].median())
    elif method == "mode":
        df[col] = df[col].fillna(df[col].mode().iloc[0])
    elif method == "constant":
        df[col] = df[col].fillna(constant)
    elif method == "ffill":
        df[col] = df[col].ffill()
    elif method == "bfill":
        df[col] = df[col].bfill()
    elif method == "drop":
        df = df.dropna(subset=[col]).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
#  METHOD & ENCODING METADATA
#  (Pure data — no Streamlit; rendered in app.py)
# ─────────────────────────────────────────────

OUTLIER_DETECTION_INFO = {
    "IQR (1.5×IQR)": {
        "full_name": "Interquartile Range Method",
        "what": (
            "Computes Q1 (25th percentile) and Q3 (75th percentile). "
            "Any value below Q1 − 1.5×IQR or above Q3 + 1.5×IQR is flagged as an outlier."
        ),
        "why": (
            "IQR is resistant to the influence of extreme values themselves, making it "
            "robust for skewed distributions. Unlike Z-score, it does not assume normality."
        ),
        "when": (
            "Use when your data is skewed, has heavy tails, or you don't know the "
            "distribution. Ideal for exploratory analysis as a first-pass outlier check."
        ),
        "pros": ["Does not assume normality", "Simple and interpretable", "Works well on skewed data"],
        "cons": ["May miss outliers in very heavy-tailed distributions", "Fixed 1.5× multiplier may be too strict/lenient"],
    },
    "Z-Score (|z| > 3)": {
        "full_name": "Standard Score / Z-Score Method",
        "what": (
            "Standardises each value as z = (x − mean) / std. "
            "Any observation with |z| > 3 is considered an outlier (covers ~99.7% of a normal distribution)."
        ),
        "why": (
            "When data is approximately normally distributed, the Z-score directly "
            "measures how many standard deviations a point lies from the mean — a natural outlier signal."
        ),
        "when": (
            "Use when your data is roughly normally distributed and sample size is large (n > 30). "
            "Avoid on highly skewed or heavy-tailed data — the mean and std are themselves distorted by outliers."
        ),
        "pros": ["Mathematically straightforward", "Widely understood", "Good for symmetric distributions"],
        "cons": ["Sensitive to the very outliers it's trying to detect", "Unreliable on skewed or non-normal data"],
    },
}

OUTLIER_ACTION_INFO = {
    "Cap / Winsorise (clip to fences)": {
        "what": "Replaces any value below the lower fence with the lower fence, and any value above the upper fence with the upper fence.",
        "when": "Use when you want to keep all rows but limit extreme influence. Common in financial and sensor data.",
        "effect": "Preserves row count. Reduces the range of the variable.",
    },
    "Fill with Mean": {
        "what": "Replaces outlier values with the column mean.",
        "when": "Use only if data is roughly symmetric (low skewness) and outliers are likely data-entry errors.",
        "effect": "Pulls extreme values to the centre. Mean is sensitive to other outliers still present.",
    },
    "Fill with Median": {
        "what": "Replaces outlier values with the column median.",
        "when": "Best choice for skewed distributions or when you suspect outliers are measurement errors.",
        "effect": "Robust replacement — median is unaffected by the outliers themselves.",
    },
    "Fill with Mode": {
        "what": "Replaces outlier values with the most frequent value in the column.",
        "when": "Rarely used for continuous numeric columns; more appropriate for discrete/ordinal data.",
        "effect": "Forces outliers to the most common value, which may not be meaningful.",
    },
    "Remove rows with outliers": {
        "what": "Drops the entire row for any observation flagged as an outlier.",
        "when": "Use only if you are confident outliers are data errors (not real but extreme events). Risky if many outliers exist.",
        "effect": "Reduces dataset size. Can introduce selection bias if outliers are genuine.",
    },
}

ENCODING_INFO = {
    "Label Encoding": {
        "what": "Assigns each unique category an integer (0, 1, 2, …) in alphabetical or appearance order.",
        "when": "Use for ordinal categories where order is meaningful (e.g. Low=0, Medium=1, High=2). "
                "Also acceptable as input to tree-based models (Decision Trees, Random Forests, XGBoost).",
        "avoid": "Avoid for nominal (unordered) categories with linear models — the integer order implies a false ranking.",
        "pros": ["Memory efficient — single column", "Works with tree models"],
        "cons": ["Implies ordinal relationship where none may exist"],
        "best_for": ["Tree-based models (RF, XGBoost, LightGBM)", "Ordinal features"],
    },
    "One-Hot Encoding": {
        "what": "Creates one binary (0/1) column per unique category. Exactly one column is 1 per row.",
        "when": "Use for nominal (unordered) categories with linear or distance-based models "
                "(Logistic Regression, SVM, KNN, Neural Networks).",
        "avoid": "Avoid when cardinality is high (>15–20 categories) — creates too many sparse columns.",
        "pros": ["No false ordinal relationship", "Works with all linear models"],
        "cons": ["Explodes dimensionality with high cardinality", "Creates sparse matrix"],
        "best_for": ["Logistic Regression", "SVM", "KNN", "Neural Networks", "Low-cardinality columns"],
    },
    "Frequency Encoding": {
        "what": "Replaces each category with its relative frequency (proportion of rows it appears in).",
        "when": "Excellent for high-cardinality columns where rarer categories should have lower weight. "
                "Works well with gradient boosting models.",
        "avoid": "Avoid if two different categories happen to have the same frequency — they become indistinguishable.",
        "pros": ["Handles high cardinality", "Preserves frequency signal", "No dimensionality increase"],
        "cons": ["Collisions when frequencies are equal", "Loses category identity"],
        "best_for": ["XGBoost", "LightGBM", "High-cardinality columns", "Memory-constrained settings"],
    },
    "Ordinal Encoding": {
        "what": "Similar to Label Encoding but allows you to explicitly specify the order of categories.",
        "when": "Use when the feature is genuinely ordinal and you know the correct order "
                "(e.g. Education: Primary < Secondary < Graduate < Postgraduate).",
        "avoid": "Avoid for nominal features — always define the order explicitly rather than relying on alphabetical defaults.",
        "pros": ["Preserves meaningful order", "Single column output"],
        "cons": ["Requires manual order specification", "Wrong order = wrong signal"],
        "best_for": ["Genuinely ordinal features", "Tree-based and linear models"],
    },
}


def get_best_encoding_recommendation(s: pd.Series) -> dict:
    """
    Analyse a categorical Series and return a dict with the recommended encoding and rationale.
    """
    n_unique = s.nunique()
    n_total  = len(s.dropna())
    pct_unique = n_unique / n_total * 100 if n_total else 0

    # Check if it looks ordinal (simple heuristic — presence of ordered keywords)
    ordinal_keywords = {"low","medium","high","small","large","poor","fair","good","excellent",
                        "never","rarely","sometimes","often","always","primary","secondary","graduate"}
    vals_lower = {str(v).lower() for v in s.dropna().unique()}
    looks_ordinal = bool(vals_lower & ordinal_keywords)

    if looks_ordinal:
        rec = "Ordinal Encoding"
        reason = ("Category values appear to have a natural order (ordinal keywords detected). "
                  "Use Ordinal Encoding and define the explicit order.")
    elif n_unique == 2:
        rec = "Label Encoding"
        reason = "Binary column (2 categories). Label Encoding (0/1) is the most efficient and widely compatible choice."
    elif n_unique <= 15:
        rec = "One-Hot Encoding"
        reason = (f"Low cardinality ({n_unique} unique values). One-Hot Encoding is safe — "
                  f"it avoids imposing false ordinal relationships and works with all model types.")
    elif n_unique <= 50:
        rec = "Frequency Encoding"
        reason = (f"Medium cardinality ({n_unique} unique values). One-Hot would create too many columns. "
                  f"Frequency Encoding is efficient and preserves the frequency signal for gradient boosting models.")
    else:
        rec = "Frequency Encoding"
        reason = (f"High cardinality ({n_unique} unique values, {pct_unique:.1f}% of rows are unique). "
                  f"One-Hot Encoding would create {n_unique} new columns — too many. "
                  f"Frequency Encoding is the safest default; consider Target Encoding if you have a label column.")

    return {
        "recommended": rec,
        "reason": reason,
        "n_unique": n_unique,
        "pct_unique": round(pct_unique, 2),
    }


# ─────────────────────────────────────────────
#  OUTLIER VISUALISATION DATA
# ─────────────────────────────────────────────

def outlier_chart_data(df: pd.DataFrame, col: str) -> dict:
    """
    Return pre-computed data needed to render outlier charts for a column.
    Returns a dict with:
      - raw: Series of non-null values
      - lo, hi: IQR fence values
      - z_lo, z_hi: Z-score fence values (mean ± 3*std)
      - is_outlier_iqr: boolean Series
      - is_outlier_z: boolean Series
      - stats: from numeric_full_stats
    """
    s = df[col].dropna()
    lo, hi, iqr_v  = iqr_fences(s)
    mean_v = s.mean(); std_v = s.std()
    z_lo   = mean_v - 3 * std_v
    z_hi   = mean_v + 3 * std_v

    z_scores       = np.abs(stats.zscore(s))
    is_out_iqr     = (s < lo) | (s > hi)
    is_out_z       = z_scores > 3

    return {
        "raw":           s,
        "lo":            lo,
        "hi":            hi,
        "z_lo":          z_lo,
        "z_hi":          z_hi,
        "is_outlier_iqr": is_out_iqr,
        "is_outlier_z":   pd.Series(is_out_z, index=s.index),
        "z_scores":       pd.Series(z_scores, index=s.index),
        "stats":          numeric_full_stats(s),
    }


def treat_outliers(df: pd.DataFrame, col: str,
                   detection: str = "iqr",
                   action: str = "cap") -> tuple[pd.DataFrame, int, str]:
    """
    Detect and treat outliers in a numeric column.

    detection: "iqr" | "zscore"
    action:    "cap" | "mean" | "median" | "mode" | "remove"

    Returns (cleaned_df, n_affected, log_message).
    """
    df = df.copy()
    s  = df[col].dropna()

    if detection == "iqr":
        lo_f, hi_f, _ = iqr_fences(s)
        mask = (df[col] < lo_f) | (df[col] > hi_f)
    else:  # zscore
        z_scores = np.abs(stats.zscore(s))
        z_idx    = s.index
        mask     = pd.Series(False, index=df.index)
        mask.loc[z_idx] = (z_scores > 3)
        lo_f = df[col].mean() - 3 * df[col].std()
        hi_f = df[col].mean() + 3 * df[col].std()

    n_affected = int(mask.sum())

    if action == "cap":
        df[col] = df[col].clip(lower=lo_f, upper=hi_f)
        msg = f"Capped {n_affected} outliers in '{col}' to [{lo_f:.3f}, {hi_f:.3f}]"
    elif action == "mean":
        fill_val = df[col].mean()
        df.loc[mask, col] = fill_val
        msg = f"Filled {n_affected} outliers in '{col}' with mean ({fill_val:.4f})"
    elif action == "median":
        fill_val = df[col].median()
        df.loc[mask, col] = fill_val
        msg = f"Filled {n_affected} outliers in '{col}' with median ({fill_val:.4f})"
    elif action == "mode":
        fill_val = df[col].mode().iloc[0]
        df.loc[mask, col] = fill_val
        msg = f"Filled {n_affected} outliers in '{col}' with mode ({fill_val:.4f})"
    elif action == "remove":
        df = df[~mask].reset_index(drop=True)
        msg = f"Removed {n_affected} rows with outliers in '{col}'"
    else:
        msg = "No action taken."

    return df, n_affected, msg


# ─────────────────────────────────────────────
#  TRANSFORMATIONS
# ─────────────────────────────────────────────

def apply_transforms(df: pd.DataFrame, col: str,
                     transforms: list) -> tuple[pd.DataFrame, dict, dict]:
    """
    Apply a list of named transforms to a column.
    Returns (df_with_new_cols, preview_dict {label: describe_series}, errors_dict {transform: error_msg}).
    """
    from sklearn.preprocessing import (
        StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
    )

    df = df.copy()
    col_data = df[[col]].dropna()
    preview  = {}
    errors   = {}

    for t in transforms:
        try:
            if t == "Log (log1p)":
                vals  = np.log1p(col_data[col])
                label = f"{col}_log1p"
            elif t == "Square Root":
                vals  = np.sqrt(col_data[col].clip(lower=0))
                label = f"{col}_sqrt"
            elif t == "Box-Cox":
                pt    = PowerTransformer(method="box-cox")
                vals  = pt.fit_transform(col_data[[col]]).flatten()
                label = f"{col}_boxcox"
            elif t == "Yeo-Johnson":
                pt    = PowerTransformer(method="yeo-johnson")
                vals  = pt.fit_transform(col_data[[col]]).flatten()
                label = f"{col}_yeojohnson"
            elif t == "Standard Scaling (Z-score)":
                sc    = StandardScaler()
                vals  = sc.fit_transform(col_data[[col]]).flatten()
                label = f"{col}_zscore"
            elif t == "MinMax Scaling [0,1]":
                sc    = MinMaxScaler()
                vals  = sc.fit_transform(col_data[[col]]).flatten()
                label = f"{col}_minmax"
            elif t == "Robust Scaling":
                sc    = RobustScaler()
                vals  = sc.fit_transform(col_data[[col]]).flatten()
                label = f"{col}_robust"
            else:
                continue

            df.loc[col_data.index, label] = vals
            preview[label] = pd.Series(vals).describe().round(4)

        except Exception as e:
            errors[t] = str(e)

    return df, preview, errors


# ─────────────────────────────────────────────
#  ENCODING PREVIEW
# ─────────────────────────────────────────────

def encoding_preview(s: pd.Series, col: str, method: str) -> pd.DataFrame:
    """
    Return a preview DataFrame showing a column alongside its encoded representation.
    method: "Label Encoding" | "One-Hot Encoding" | "Frequency Encoding" | "Ordinal Encoding"
    """
    from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

    s = s.fillna("Missing")

    if method == "Label Encoding":
        le = LabelEncoder()
        return pd.DataFrame({col: s, "Encoded": le.fit_transform(s)})

    elif method == "One-Hot Encoding":
        ohe = pd.get_dummies(s, prefix=col).astype(int)
        ohe.insert(0, col, s.values)
        return ohe

    elif method == "Frequency Encoding":
        freq_map = s.value_counts(normalize=True)
        return pd.DataFrame({col: s, "Freq_Encoded": s.map(freq_map).round(4)})

    else:  # Ordinal Encoding

        s = s.fillna("Missing").astype(str)

        cats = sorted(s.unique())

        oe = OrdinalEncoder(categories=[cats])

        encoded = oe.fit_transform(
            s.to_numpy().reshape(-1, 1)
        ).flatten()

    return pd.DataFrame({
        col: s,
        "Ordinal": encoded,
    })

def apply_encoding(df: pd.DataFrame, col: str, method: str,
                   ordinal_order: list | None = None) -> tuple[pd.DataFrame, str]:
    """
    Apply encoding to a column and return (new_df, log_message).
    The original column is kept; encoded column(s) are added alongside it.
    method: "Label Encoding" | "One-Hot Encoding" | "Frequency Encoding" | "Ordinal Encoding"
    ordinal_order: explicit list of categories in ascending order (for Ordinal Encoding).
    """
    from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

    df = df.copy()
    s  = df[col].fillna("Missing")

    if method == "Label Encoding":
        le = LabelEncoder()
        df[f"{col}_label_enc"] = le.fit_transform(s.astype(str))
        msg = f"Label Encoded '{col}' → '{col}_label_enc'"

    elif method == "One-Hot Encoding":
        ohe = pd.get_dummies(s, prefix=col).astype(int)
        # drop original col, add new binary columns
        df = df.drop(columns=[col])
        for c in ohe.columns:
            df[c] = ohe[c].values
        msg = f"One-Hot Encoded '{col}' → {list(ohe.columns)}"

    elif method == "Frequency Encoding":
        freq_map = s.value_counts(normalize=True)
        df[f"{col}_freq_enc"] = s.map(freq_map).round(4)
        msg = f"Frequency Encoded '{col}' → '{col}_freq_enc'"

    else:  # Ordinal Encoding
        s_str = s.astype(str)
        cats  = ordinal_order if ordinal_order else sorted(s_str.unique())
        oe    = OrdinalEncoder(categories=[cats])
        df[f"{col}_ordinal_enc"] = oe.fit_transform(
            s_str.to_numpy().reshape(-1, 1)
        ).flatten()
        msg = f"Ordinal Encoded '{col}' → '{col}_ordinal_enc'"

    return df, msg


# ─────────────────────────────────────────────
#  RECOMMENDATIONS ENGINE
# ─────────────────────────────────────────────

def build_recommendations(df: pd.DataFrame,
                           num_cols: list,
                           cat_cols: list) -> dict:
    """
    Analyse the DataFrame and return a structured recommendations dict.

    Returns:
    {
      "duplicates":   {"count": int, "pct": float},
      "missing":      [{"col": str, "pct": float, "is_numeric": bool, "skewness": float|None}],
      "skewness":     [{"col": str, "skewness": float}],
      "outliers":     [{"col": str, "n_iqr": int, "pct": float, "lo": float, "hi": float, "n_z": int}],
      "cardinality":  [{"col": str, "n_unique": int}],
      "low_variance": [{"col": str, "std": float, "cv_pct": float|None}],
      "imbalance":    [{"col": str, "dominant_val": str, "dominant_pct": float}],
      "high_corr":    [{"col_a": str, "col_b": str, "r": float}],
      "checklist":    [(label, bool)],
      "scorecard":    {"score": int, "grade": str, "critical": int, "warning": int},
    }
    """
    all_stats    = {col: numeric_full_stats(df[col]) for col in num_cols}
    miss_pct     = (df.isna().mean() * 100).round(2)
    dupes_n      = duplicate_rows(df)
    dupes_pct    = round(dupes_n / len(df) * 100, 2) if len(df) else 0

    # ── Missing ──────────────────────────────────────────────
    missing_recs = []
    for col in df.columns:
        pct = float(miss_pct.get(col, 0))
        if pct > 0:
            skewness = all_stats.get(col, {}).get("skewness") if col in num_cols else None
            missing_recs.append({
                "col": col, "pct": pct,
                "is_numeric": col in num_cols,
                "skewness": skewness,
            })

    # ── Skewness ─────────────────────────────────────────────
    skew_recs = [
        {"col": col, "skewness": all_stats[col]["skewness"]}
        for col in num_cols
        if abs(all_stats.get(col, {}).get("skewness", 0)) > 1
    ]

    # ── Outliers ─────────────────────────────────────────────
    outlier_recs = []
    for col in num_cols:
        st = all_stats.get(col, {})
        n_iqr = st.get("outliers_iqr", 0)
        if n_iqr > 0:
            pct_out = round(n_iqr / df[col].dropna().shape[0] * 100, 2)
            outlier_recs.append({
                "col": col, "n_iqr": n_iqr, "pct": pct_out,
                "lo":  st.get("lower_fence", 0),
                "hi":  st.get("upper_fence", 0),
                "n_z": st.get("outliers_z", 0),
            })

    # ── Cardinality ───────────────────────────────────────────
    card_recs = [
        {"col": col, "n_unique": df[col].nunique()}
        for col in cat_cols
        if df[col].nunique() > 50 or df[col].nunique() == 1
    ]

    # ── Low / Zero Variance ───────────────────────────────────
    low_var_recs = []
    for col in num_cols:
        st = all_stats.get(col, {})
        std = st.get("std", 1)
        cv  = st.get("cv_pct")
        if std < 1e-6 or (cv is not None and abs(cv) < 1):
            low_var_recs.append({"col": col, "std": std, "cv_pct": cv})

    # ── Class Imbalance ───────────────────────────────────────
    imbalance_recs = []
    for col in cat_cols:
        vc  = df[col].value_counts(normalize=True) * 100
        dom = vc.iloc[0]
        if dom > 80:
            imbalance_recs.append({
                "col": col,
                "dominant_val": str(vc.index[0]),
                "dominant_pct": round(dom, 2),
            })

    # ── Multicollinearity ─────────────────────────────────────
    high_corr_recs = []
    if len(num_cols) >= 2:
        corr_mat = df[num_cols].corr().abs()
        for i in range(len(corr_mat.columns)):
            for j in range(i + 1, len(corr_mat.columns)):
                r = corr_mat.iloc[i, j]
                if r > 0.85:
                    high_corr_recs.append({
                        "col_a": corr_mat.columns[i],
                        "col_b": corr_mat.columns[j],
                        "r":     round(r, 3),
                    })

    # ── Scorecard ─────────────────────────────────────────────
    n_critical = (
        (1 if dupes_pct > 5 else 0)
        + sum(1 for r in missing_recs if r["pct"] > 30)
        + sum(1 for r in outlier_recs if r["pct"] >= 5)
        + sum(1 for r in card_recs if r["n_unique"] == 1)
        + sum(1 for r in imbalance_recs if r["dominant_pct"] >= 90)
        + len(low_var_recs)
    )
    n_warning = (
        (1 if 0 < dupes_pct <= 5 else 0)
        + sum(1 for r in missing_recs if 0 < r["pct"] <= 30)
        + len(skew_recs)
        + sum(1 for r in outlier_recs if r["pct"] < 5)
        + sum(1 for r in card_recs if r["n_unique"] > 50)
        + sum(1 for r in imbalance_recs if r["dominant_pct"] < 90)
        + len(high_corr_recs)
    )
    score = max(0, 100 - n_critical * 15 - n_warning * 5)
    grade = "A" if score >= 90 else "B" if score >= 75 else "C" if score >= 60 else "D"

    # ── EDA Checklist ─────────────────────────────────────────
    checklist = [
        ("Check shape & dtypes",                     True),
        ("Handle missing values",                    not (miss_pct > 0).any()),
        ("Remove duplicate rows",                    dupes_n == 0),
        ("Detect & treat outliers (IQR / Z-score)",  all(all_stats.get(c, {}).get("outliers_iqr", 0) == 0 for c in num_cols)),
        ("Check skewness → transform if needed",     all(abs(all_stats.get(c, {}).get("skewness", 0)) < 1 for c in num_cols)),
        ("Verify normality (Q-Q plot, Shapiro-Wilk)", all(all_stats.get(c, {}).get("is_normal", True) for c in num_cols)),
        ("Encode categorical columns",               not bool(cat_cols)),
        ("Check correlation / multicollinearity",    True),
        ("Scale/normalise features",                 False),
        ("Check class balance (if classification)",  True),
        ("Export cleaned dataset (.pkl / .csv)",     False),
    ]

    return {
        "duplicates":   {"count": dupes_n, "pct": dupes_pct},
        "missing":      missing_recs,
        "skewness":     skew_recs,
        "outliers":     outlier_recs,
        "cardinality":  card_recs,
        "low_variance": low_var_recs,
        "imbalance":    imbalance_recs,
        "high_corr":    high_corr_recs,
        "checklist":    checklist,
        "scorecard":    {
            "score":    score,
            "grade":    grade,
            "critical": n_critical,
            "warning":  n_warning,
        },
        "_all_stats":   all_stats,   # pass-through for reuse in UI
        "_miss_pct":    miss_pct,
    }
