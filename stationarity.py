from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from statsmodels.tsa.stattools import adfuller


@dataclass(frozen=True)
class StationarityResult:
    series_name: str
    adf_p: float
    kpss_p: float | None
    is_stationary: bool
    integration_order: int


def _adf_pvalue(series: pd.Series, regression: str = 'c', autolag: str = 'aic', maxlag: int = None) -> float:
    try:
        res = adfuller(series, autolag=autolag, regression=regression, maxlag=maxlag)
        return float(res[1])
    except Exception:
        return 1.0


def stationarity_adf_kpss(series: pd.Series, name: str) -> StationarityResult:
    s = pd.to_numeric(series, errors='coerce').dropna()
    print(f"[DEBUG] {name}: исходных наблюдений = {len(s)}")

    # Уровни (только константа, AIC)
    p_level = _adf_pvalue(s, regression='c', autolag='aic')
    print(f"[DEBUG] {name}: p_level (c, AIC) = {p_level:.6f}")
    if p_level < 0.05:
        print(f"[DEBUG] {name}: стационарен на уровнях → I(0)")
        return StationarityResult(name, p_level, None, True, 0)

    # Первые разности (константа + тренд, BIC, maxlag=5)
    s_diff1 = s.diff().dropna()
    print(f"[DEBUG] {name}: diff1 наблюдений = {len(s_diff1)}")
    p_diff1 = _adf_pvalue(s_diff1, regression='ct', autolag='bic', maxlag=5)
    print(f"[DEBUG] {name}: p_diff1 (ct, BIC, maxlag=5) = {p_diff1:.6f}")
    if p_diff1 < 0.05:
        print(f"[DEBUG] {name}: стационарен на первых разностях → I(1)")
        return StationarityResult(name, p_diff1, None, True, 1)

    # Вторые разности (только константа, AIC)
    s_diff2 = s_diff1.diff().dropna()
    print(f"[DEBUG] {name}: diff2 наблюдений = {len(s_diff2)}")
    p_diff2 = _adf_pvalue(s_diff2, regression='c', autolag='aic')
    print(f"[DEBUG] {name}: p_diff2 (c, AIC) = {p_diff2:.6f}")
    is_stat2 = p_diff2 < 0.05
    order = 2 if not is_stat2 else 2
    print(f"[DEBUG] {name}: итоговый порядок интеграции I({order})")
    return StationarityResult(name, p_diff2, None, is_stat2, order)