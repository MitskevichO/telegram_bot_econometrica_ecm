from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

@dataclass(frozen=True)
class ForecastResult:
    x_values: list[float]
    y_hat: list[float]
    lower: list[float] | None
    upper: list[float] | None
    mape: float | None


def point_forecast_ols_with_x(df: pd.DataFrame, y_col: str, x_col: str, x_future: Sequence[float]) -> ForecastResult:
    """Точечный прогноз на основе линейной регрессии (МНК) y ~ x."""
    d = df[[y_col, x_col]].dropna()
    X = np.column_stack((np.ones(len(d)), d[x_col].values))
    y = d[y_col].values
    m = sm.OLS(y, X).fit()

    X_future = np.column_stack((np.ones(len(x_future)), x_future))
    y_hat = m.predict(X_future).tolist()

    y_fit = m.predict(X)
    mape = float(np.mean(np.abs((y - y_fit) / np.clip(np.abs(y), 1e-9, None))) * 100.0)

    return ForecastResult(
        x_values=list(x_future),
        y_hat=y_hat,
        lower=None,
        upper=None,
        mape=mape,
    )
def forecast_with_ci_ols_with_x(
    df: pd.DataFrame, y_col: str, x_col: str, x_future: Sequence[float], ci_level: int
) -> ForecastResult:
    """Прогноз с доверительным интервалом на основе линейной регрессии (МНК) y ~ x."""
    d = df[[y_col, x_col]].dropna()
    X = np.column_stack((np.ones(len(d)), d[x_col].values))
    y = d[y_col].values
    m = sm.OLS(y, X).fit()

    X_future = np.column_stack((np.ones(len(x_future)), x_future))
    pred = m.get_prediction(X_future)
    alpha = 1 - ci_level / 100.0
    frame = pred.summary_frame(alpha=alpha)

    y_hat = frame["mean"].tolist()
    lower = frame["obs_ci_lower"].tolist()
    upper = frame["obs_ci_upper"].tolist()

    y_fit = m.predict(X)
    mape = float(np.mean(np.abs((y - y_fit) / np.clip(np.abs(y), 1e-9, None))) * 100.0)

    return ForecastResult(
        x_values=list(x_future),
        y_hat=y_hat,
        lower=lower,
        upper=upper,
        mape=mape,
    )


def forecast_ecm(
    df: pd.DataFrame,
    y_col: str,
    x_col: str,
    x_future: Sequence[float],
    lags_dy: int = 0,
    ci_level: float = 95.0,
) -> ForecastResult:
    """
    Прогноз на основе ECM с приближёнными доверительными интервалами.
    Доверительный интервал строится как +/- t * sigma_u * sqrt(h),
    где h – номер шага прогноза (от 1 до n_future).
    Поддерживаются модели с lags_dy = 0 или 1.
    """
    d = df[[y_col, x_col]].copy().dropna()
    if len(d) < 20:
        raise ValueError("Too few observations for ECM forecast")

    # ----- 1. Долгосрочное коинтеграционное уравнение -----
    X_lr = sm.add_constant(d[x_col])
    lr_model = sm.OLS(d[y_col], X_lr).fit()
    beta0 = lr_model.params["const"]
    beta1 = lr_model.params[x_col]
    d["ect"] = d[y_col] - (beta0 + beta1 * d[x_col])

    # ----- 2. Краткосрочная ECM -----
    dy = d[y_col].diff()
    dx = d[x_col].diff()
    ect_lag = d["ect"].shift(1)

    # Формируем матрицу регрессоров ECM
    ecm_data = pd.DataFrame({"dy": dy, "dx": dx, "ect_lag": ect_lag})
    if lags_dy >= 1:
        ecm_data["dy_lag1"] = dy.shift(1)
    ecm_data = ecm_data.dropna()

    # Список регрессоров (без dy)
    regressors = ["dx", "ect_lag"]
    if lags_dy >= 1:
        regressors.append("dy_lag1")
    X_ecm = sm.add_constant(ecm_data[regressors])
    ecm_model = sm.OLS(ecm_data["dy"], X_ecm).fit()

    # Коэффициенты
    const = ecm_model.params.get("const", 0.0)
    gamma = ecm_model.params.get("dx", 0.0)
    alpha = ecm_model.params.get("ect_lag", 0.0)
    delta = ecm_model.params.get("dy_lag1", 0.0) if lags_dy >= 1 else 0.0

    # Стандартная ошибка регрессии (сигма ошибки)
    sigma_u = np.std(ecm_model.resid, ddof=len(ecm_model.params))

    # ----- 3. Точечный рекурсивный прогноз -----
    # Начальные условия (последние фактические значения)
    last_y = d[y_col].iloc[-1]
    last_x = d[x_col].iloc[-1]
    last_ect = d["ect"].iloc[-1]
    # Для lags_dy=1 нужно последнее фактическое изменение dy
    if lags_dy >= 1:
        # dy на последнем историческом периоде (разность между последним и предпоследним Y)
        last_dy = dy.iloc[-1] if not pd.isna(dy.iloc[-1]) else 0.0
    else:
        last_dy = 0.0

    y_hat = []
    # Для хранения предсказанных dy (нужны для следующих шагов при lags_dy=1)
    pred_dy = []

    for i, x_next in enumerate(x_future):
        dx_next = x_next - last_x

        # Расчёт прогнозного изменения
        dy_pred = const + gamma * dx_next + alpha * last_ect
        if lags_dy >= 1:
            dy_pred += delta * last_dy

        y_next = last_y + dy_pred
        y_hat.append(y_next)

        # Обновление для следующего шага
        last_y = y_next
        last_x = x_next
        last_ect = y_next - (beta0 + beta1 * x_next)
        if lags_dy >= 1:
            last_dy = dy_pred   # для следующего шага используем предсказанное изменение

    # ----- 4. Приближённые доверительные интервалы -----
    alpha_ci = 1 - ci_level / 100.0
    # Критическое значение t-распределения (степени свободы = остаточные)
    df_resid = len(ecm_model.resid) - len(ecm_model.params)
    t_crit = stats.t.ppf(1 - alpha_ci / 2, df=df_resid)

    lower = []
    upper = []
    for h in range(1, len(x_future) + 1):
        half_width = t_crit * sigma_u * np.sqrt(h)
        lower.append(y_hat[h-1] - half_width)
        upper.append(y_hat[h-1] + half_width)

    # ----- 5. Расчёт MAPE на исторических данных (одношаговые прогнозы) -----
    # Прогнозируем Y(t) начиная с момента, когда доступны все лаги
    # Используем рекурсивный детерминированный прогноз на истории
    n = len(d)
    y_true_hist = d[y_col].values
    x_hist = d[x_col].values
    ect_hist = d["ect"].values

    pred_y_hist = []
    actual_y_hist = []
    # Определяем стартовый индекс (чтобы были все лаги)
    start_idx = 1  # для lags_dy=0 достаточно одного предыдущего периода
    if lags_dy >= 1:
        start_idx = 2  # нужны dy_{t-1} и ect_{t-1}

    for t in range(start_idx, n):
        # Используем фактические значения для лагов
        dx_t = x_hist[t] - x_hist[t-1]
        ect_t_1 = ect_hist[t-1]
        dy_t_1 = y_true_hist[t-1] - y_true_hist[t-2] if t-2 >= 0 else 0.0

        dy_pred = const + gamma * dx_t + alpha * ect_t_1
        if lags_dy >= 1:
            dy_pred += delta * dy_t_1

        y_pred = y_true_hist[t-1] + dy_pred
        pred_y_hist.append(y_pred)
        actual_y_hist.append(y_true_hist[t])

    if actual_y_hist:
        y_true_arr = np.array(actual_y_hist)
        y_pred_arr = np.array(pred_y_hist)
        mape_ecm = float(np.mean(np.abs((y_true_arr - y_pred_arr) / np.clip(np.abs(y_true_arr), 1e-9, None))) * 100.0)
    else:
        mape_ecm = None

    return ForecastResult(
        x_values=list(x_future),
        y_hat=y_hat,
        lower=lower,
        upper=upper,
        mape=mape_ecm,
    )