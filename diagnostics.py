from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import acorr_breusch_godfrey

@dataclass(frozen=True)
class TestResult:
    name: str
    p_value: float
    ok: bool


def white_test(residuals: pd.Series, exog: pd.DataFrame) -> TestResult:
    res = pd.to_numeric(residuals, errors="coerce").dropna()
    #только те строки из матрицы регрессоров, которые соответствуют оставшимся наблюдениям остатков
    X = exog.loc[res.index].copy()

    # Добавляем столбец константы для свободного члена во вспомогательной регрессии
    X_const = sm.add_constant(X, has_constant="add")
    # Список, в который будут собраны все группы регрессоров: константа, квадраты, попарные произведения
    X_terms = [X_const]
    # Для каждого исходного регрессора добавляем его квадрат (столбец вида "x^2")
    for col in X.columns:
        X_terms.append(pd.DataFrame({f"{col}^2": X[col] ** 2}))
    # Если исходных регрессоров больше одного, добавляем все попарные произведения (x1*x2, x1*x3, ...)
    if X.shape[1] > 1:
        for i in range(X.shape[1]):
            for j in range(i + 1, X.shape[1]):
                col_i = X.columns[i]
                col_j = X.columns[j]
                X_terms.append(pd.DataFrame({f"{col_i}*{col_j}": X[col_i] * X[col_j]}))

    Z = pd.concat(X_terms, axis=1)
    #  Удаляем возможные дубликаты столбцов (например, если константа добавилась несколько раз)
    Z = Z.loc[:, ~Z.columns.duplicated()]
    # Зависимая переменная во вспомогательной регрессии – квадраты остатков исходной модели
    resid_sq = res ** 2
    model = sm.OLS(resid_sq, Z, missing='drop').fit()
    n = int(model.nobs)
    r2 = model.rsquared
    lm_stat = n * r2
    # Число степеней свободы – количество неконстантных регрессоров во вспомогательной модели
    df = int(model.df_model)
    p = 1 - stats.chi2.cdf(lm_stat, df)
    # Возвращаем результат: если p > 0.05 – гомоскедастичность не отвергается (остатки устойчивы)
    return TestResult(name="White", p_value=p, ok=p > 0.05)


def breusch_godfrey_lm(model: sm.regression.linear_model.RegressionResultsWrapper, nlags: int = 1) -> TestResult:
    lm_stat, lm_p, _, _ = acorr_breusch_godfrey(model, nlags=nlags)
    p = float(lm_p)
    return TestResult(name="Breusch-Godfrey", p_value=p, ok=p > 0.05)


def breusch_godfrey_resid_exog(residuals: pd.Series, exog: pd.DataFrame, nlags: int = 1) -> TestResult:
    """
    Тест Бройша-Годфри для автокорреляции остатков.
    Используется, когда у нас нет полной модели, но есть остатки и регрессоры.
    """
    resid = pd.to_numeric(residuals, errors="coerce").dropna()
    # Создаём лаги остатков
    n = len(resid)
    lagged = pd.DataFrame({f"resid_lag{i+1}": resid.shift(i+1) for i in range(nlags)})
    # Объединяем с экзогенными переменными (должны быть на тех же индексах)
    # Приводим exog к индексам остатков
    exog_aligned = exog.loc[resid.index]
    data = pd.concat([resid, exog_aligned, lagged], axis=1).dropna()
    y = data.iloc[:, 0]  # остатки
    X = data.iloc[:, 1:] # exog + лаги
    aux_model = sm.OLS(y, sm.add_constant(X)).fit()
    lm_stat = aux_model.nobs * aux_model.rsquared
    df = nlags
    p = 1 - stats.chi2.cdf(lm_stat, df)
    return TestResult(name="Breusch-Godfrey", p_value=p, ok=p > 0.05)


def chow_test_simple(df: pd.DataFrame, y_col: str, x_col: str, break_index: int) -> TestResult:
    """
    """
    d = df[[y_col, x_col]].copy().dropna()
    if break_index <= 5 or break_index >= len(d) - 2:
        raise ValueError("Break index too close to edges")

    d1 = d.iloc[:break_index]
    d2 = d.iloc[break_index:]

    def _fit(dd: pd.DataFrame):
        X = sm.add_constant(dd[x_col])
        return sm.OLS(dd[y_col], X).fit()

    m = _fit(d)
    m1 = _fit(d1)
    m2 = _fit(d2)

    k = int(m.df_model) + 1  # parameters count incl. constant
    rss_pooled = float(np.sum(m.resid**2))
    rss_1 = float(np.sum(m1.resid**2))
    rss_2 = float(np.sum(m2.resid**2))
    n1, n2 = len(d1), len(d2)

    num = (rss_pooled - (rss_1 + rss_2)) / k
    den = (rss_1 + rss_2) / (n1 + n2 - 2 * k)
    F = num / den
    p = float(1 - stats.f.cdf(F, k, n1 + n2 - 2 * k))
    return TestResult(name="Chow", p_value=p, ok=p > 0.05)