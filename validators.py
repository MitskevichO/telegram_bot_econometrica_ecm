from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    n_obs: int
    period_start: str
    period_end: str
    missing_filled: int
    freq: str | None
    date_col: str
    numeric_cols: list[str]

    def model_dump(self) -> dict:
        return {
            "ok": self.ok,
            "n_obs": self.n_obs,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "missing_filled": self.missing_filled,
            "freq": self.freq,
            "date_col": self.date_col,
            "numeric_cols": self.numeric_cols,
        }


def load_table_from_bytes(file_bytes: bytes, filename: str) -> pd.DataFrame:
    name = filename.lower()
    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes))
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(io.BytesIO(file_bytes))
    raise ValueError("К сожалению, этот формат файла я не поддерживаю")


def is_year_column(series: pd.Series) -> bool:
    """Проверяет, содержит ли колонка целые числа в диапазоне 1900–2100 (годы)."""
    try:
        # Приводим к числовому типу, если возможно
        nums = pd.to_numeric(series, errors='coerce')
        if nums.isna().all():
            return False
        # Проверяем, что все значения целые (отклонение от целого < 1e-6)
        if np.allclose(nums, nums.round(), atol=1e-6):
            int_vals = nums.round().astype(int)
            return int_vals.between(1900, 2100).all()
        return False
    except:
        return False


def _detect_date_col(df: pd.DataFrame) -> str:
    # 1. По ключевым словам
    for c in df.columns:
        low = str(c).lower()
        if low in {"date", "ds", "time", "period", "год", "year"}:
            return str(c)
    # 2. Пробуем все колонки: конвертируем в datetime
    for c in df.columns:
        try:
            test = pd.to_datetime(df[c], errors='coerce')
            if test.notna().sum() > len(df) * 0.8:
                return str(c)
        except:
            continue
    # 3. Ищем колонку, которая содержит целые числа в диапазоне годов
    for c in df.columns:
        if is_year_column(df[c]):
            return str(c)
    raise ValueError(
        "Не удалось определить колонку с датами. Убедитесь, что она называется 'date' или 'year', либо содержит годы (2000,2001...).")


def validate_table(df: pd.DataFrame) -> ValidationResult:
    if df.shape[1] < 3:
        raise ValueError("Need date column + at least 2 numeric columns")

    df = df.copy()
    date_col = _detect_date_col(df)

    # Преобразование дат
    if is_year_column(df[date_col]):
        # Годы (2000, 2001, ...)
        df[date_col] = pd.to_datetime(df[date_col], format='%Y', errors='coerce')
    else:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    # Удаляем строки, где дата не распозналась
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    numeric_cols = [c for c in df.columns if c != date_col]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    numeric_cols = [c for c in numeric_cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) < 2:
        raise ValueError("Need at least 2 numeric columns")

    # Интерполяция пропусков
    before_missing = int(df[numeric_cols].isna().sum().sum())
    if before_missing:
        df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")
    after_missing = int(df[numeric_cols].isna().sum().sum())
    missing_filled = before_missing - after_missing

    # Частота
    freq = None
    try:
        freq = pd.infer_freq(df[date_col])
    except Exception:
        freq = None

    n_obs = int(df.shape[0])
    period_start = str(df[date_col].iloc[0].date())
    period_end = str(df[date_col].iloc[-1].date())

    # Очистка
    df = df[[date_col] + numeric_cols]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if df.shape[0] < 20:
        raise ValueError("Too few observations (need at least 20 rows after cleaning)")

    return ValidationResult(
        ok=True,
        n_obs=n_obs,
        period_start=period_start,
        period_end=period_end,
        missing_filled=missing_filled,
        freq=freq,
        date_col=date_col,
        numeric_cols=numeric_cols[:5],
    )