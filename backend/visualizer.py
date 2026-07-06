import matplotlib
matplotlib.use('Agg')  # Headless backend for web server
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
import base64


def plot_missing_values(df):
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0].sort_values(ascending=False).head(15)

    if null_counts.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=null_counts.index, y=null_counts.values, ax=ax)
    ax.set_title("Top Columns with Missing Values")
    ax.set_ylabel("Missing Count")
    ax.set_xlabel("Columns")
    ax.tick_params(axis="x", rotation=75)
    plt.tight_layout()

    return fig


def plot_target_distribution(df, target_col):
    if target_col not in df.columns:
        return None

    target_series = df[target_col].dropna()

    if target_series.nunique() > 20:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x=target_series.astype(str), ax=ax)
    ax.set_title(f"Target Distribution: {target_col}")
    ax.set_xlabel(target_col)
    ax.set_ylabel("Count")
    plt.tight_layout()

    return fig


def plot_correlation_heatmap(df, column_types):
    numerical_cols = column_types["numerical"]

    if len(numerical_cols) < 2:
        return None

    selected_cols = numerical_cols[:12]  # keep heatmap readable
    corr_df = df[selected_cols].copy()
    corr_df = corr_df.apply(pd.to_numeric, errors="coerce")

    corr_matrix = corr_df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, cmap="coolwarm", annot=False, ax=ax)
    ax.set_title("Correlation Heatmap (Numerical Features)")
    plt.tight_layout()

    return fig


def fig_to_base64(fig):
    if fig is None:
        return None
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        return img_str
    except Exception as e:
        print(f"Error converting figure to base64: {e}")
        return None