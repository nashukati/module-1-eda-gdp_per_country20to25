"""
Quick EDA helper functions.
Note: Each chart is created in its own figure (no subplots).
"""
from typing import List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def basic_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact table with dtype, #nulls, %nulls, n_unique, example values."""
    rows = []
    n = len(df)
    for col in df.columns:
        nulls = df[col].isna().sum()
        nunique = df[col].nunique(dropna=True)
        example = df[col].dropna().head(3).tolist()
        rows.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "nulls": int(nulls),
            "null_pct": round(100 * nulls / max(n, 1), 2),
            "n_unique": int(nunique),
            "examples": example
        })
    return pd.DataFrame(rows).sort_values(by=["null_pct","n_unique"], ascending=[False, True]).reset_index(drop=True)

def plot_numeric_distribution(df: pd.DataFrame, col: str) -> None:
    """Histogram for a numeric column."""
    plt.figure()
    df[col].dropna().plot(kind="hist", bins=30, alpha=0.8)
    plt.title(f"Distribution: {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def plot_categorical_counts(df: pd.DataFrame, col: str, top: Optional[int]=20) -> None:
    """Bar chart of counts for a categorical column."""
    plt.figure()
    vc = df[col].astype("category").value_counts().head(top)
    vc.plot(kind="bar")
    plt.title(f"Counts: {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def correlation_heatmap(df: pd.DataFrame, numeric_only: bool=True) -> None:
    """Correlation heatmap for numeric columns."""
    if numeric_only:
        corr = df.select_dtypes(include=[np.number]).corr()
    else:
        corr = df.corr(numeric_only=True)
    plt.figure()
    sns.heatmap(corr, annot=False)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

def detect_outliers_iqr(series: pd.Series, k: float=1.5) -> pd.Series:
    """Return a boolean mask of outliers using IQR rule."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (series < lower) | (series > upper)
