from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass(frozen=True)
class ECMResult:
    y: str
    x: str
    alpha: float
    params: dict[str, float]
    pvalues: dict[str, float]
    free_coef: float
    r2: float


def build_ecm(df: pd.DataFrame, y_col: str, x_col: str, lags_dy: int = 1):
    """
    Simple ECM:
    1) Long-run: y = b0 + а x + e
    2) Short-run: dy = alpha * ect(-1) + gamma * dx + sum delta_i * dy(-i) + u
    Returns (ECMResult, regression_results) where regression_results is the fitted OLS.
    """
    d = df[[y_col, x_col]].copy()
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d = d.dropna()
    if len(d) < 20:
        raise ValueError("Слишком мало наблюдений для построения ECM (меньше 20)")

    # долгосрочное уравнение
    X_lr = sm.add_constant(d[x_col])
    lr = sm.OLS(d[y_col], X_lr).fit()
    ect = lr.resid.shift(1)

    dy = d[y_col].diff()
    dx = d[x_col].diff()

    reg = pd.DataFrame({"dy": dy, "dx": dx, "ect": ect})
    for i in range(1, lags_dy + 1):
        reg[f"dy_lag{i}"] = dy.shift(i)
    reg = reg.dropna()

    X_sr = sm.add_constant(reg.drop(columns=["dy"]))
    sr = sm.OLS(reg["dy"], X_sr).fit()

    free_coef = float(sr.params.get("const", 0.0))
    alpha = float(sr.params.get("ect", np.nan))
    gamma = float(sr.params.get("dx", np.nan))
    delta = float(sr.params.get("dy_lag1", np.nan)) if lags_dy >= 1 else 0.0

    params = {"const": free_coef, "dx": gamma, "ect": alpha}
    pvalues = {"const": float(sr.pvalues.get("const", 1.0)),
               "dx": float(sr.pvalues.get("dx", 1.0)),
               "ect": float(sr.pvalues.get("ect", 1.0))}
    if lags_dy >= 1:
        params["dy_lag1"] = delta
        pvalues["dy_lag1"] = float(sr.pvalues.get("dy_lag1", 1.0))

    ecm_res = ECMResult(
        y=y_col,
        x=x_col,
        alpha=alpha,
        params=params,
        pvalues=pvalues,
        free_coef=free_coef,
        r2=float(sr.rsquared),
    )
    return ecm_res, sr