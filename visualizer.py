from __future__ import annotations

import io

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def _setup() -> None:
    matplotlib.use("Agg", force=True)
    sns.set_theme(style="whitegrid", palette="colorblind")


def fig_to_png_bytes(fig: plt.Figure, dpi: int = 300) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def plot_timeseries(df: pd.DataFrame, date_col: str, y_col: str, x_col: str | None = None) -> bytes:
    _setup()
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(df[date_col], df[y_col], label=y_col, linewidth=2)
    if x_col:
        ax.plot(df[date_col], df[x_col], label=x_col, linewidth=2, alpha=0.85)
    ax.set_title("Временные ряды")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Значение")
    ax.legend()
    return fig_to_png_bytes(fig)


def plot_stationarity_panels(df: pd.DataFrame, date_col: str, series_col: str) -> bytes:
    _setup()
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 10), sharex=True)
    axes[0].plot(df[date_col], df[series_col], linewidth=2)
    axes[0].set_title(f"{series_col}: уровни")
    axes[1].plot(df[date_col], df[series_col].diff(), linewidth=2)
    axes[1].set_title(f"{series_col}: первая разность")
    axes[2].plot(df[date_col], df[series_col].diff().diff(), linewidth=2)
    axes[2].set_title(f"{series_col}: вторая разность")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig_to_png_bytes(fig)



def plot_corr_heatmap(df: pd.DataFrame, cols: list[str]) -> bytes:
    _setup()
    corr = df[cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, ax=ax, annot=True, fmt=".2f", cmap="vlag", square=True, cbar=True)
    ax.set_title("Коррелограмма (матрица корреляций)")
    return fig_to_png_bytes(fig)


def plot_acf_pacf_panels(series: pd.Series, title: str) -> bytes:
    _setup()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    s = pd.to_numeric(series, errors="coerce").dropna()
    plot_acf(s, ax=axes[0], lags=min(40, max(10, len(s) // 3)))
    plot_pacf(s, ax=axes[1], lags=min(40, max(10, len(s) // 3)), method="ywm")
    axes[0].set_title("Автокорреляционная функция (ACF)")
    axes[1].set_title("Частная автокорреляционная функция (PACF)")
    fig.suptitle(title)
    fig.tight_layout()
    return fig_to_png_bytes(fig)


def plot_residuals_panels(residuals: pd.Series, title: str = "Остатки") -> bytes:
    _setup()
    r = pd.to_numeric(residuals, errors="coerce").dropna()
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
    axes[0].plot(r.values, linewidth=1.5)
    axes[0].set_title(f"{title}: временной ряд")
    sns.histplot(r, kde=True, ax=axes[1])
    axes[1].set_title(f"{title}: гистограмма ")
    fig.tight_layout()
    return fig_to_png_bytes(fig)


def plot_fitted_vs_actual(y_true: pd.Series, y_fit: pd.Series, title: str) -> bytes:
    _setup()
    yt = pd.to_numeric(y_true, errors="coerce")
    yf = pd.to_numeric(y_fit, errors="coerce")
    d = pd.DataFrame({"actual": yt, "fitted": yf}).dropna()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(d["actual"], d["fitted"], alpha=0.7)
    lo = float(min(d["actual"].min(), d["fitted"].min()))
    hi = float(max(d["actual"].max(), d["fitted"].max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Фактические")
    ax.set_ylabel("Прогнозные")
    ax.grid(True, alpha=0.3)
    return fig_to_png_bytes(fig)
def plot_residuals_series(residuals: pd.Series, title: str = "Остатки") -> bytes:
    """Только временной ряд остатков """
    _setup()
    r = pd.to_numeric(residuals, errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(r.values, linewidth=1.5, color='blue')
    ax.set_title(title)
    ax.set_xlabel("Наблюдения")
    ax.set_ylabel("Остатки")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig_to_png_bytes(fig)