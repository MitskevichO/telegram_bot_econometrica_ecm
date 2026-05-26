from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


@dataclass(frozen=True)
class EngleGrangerResult:
    y: str
    x: str
    adf_p_resid: float
    cointegrated: bool
    beta0: float
    beta1: float


def engle_granger_test(y: pd.Series, x: pd.Series, y_name: str, x_name: str) -> EngleGrangerResult:
    # Принудительно преобразуем ряды в числовой тип (нечисловые значения станут NaN)
    yv = pd.to_numeric(y, errors="coerce")
    xv = pd.to_numeric(x, errors="coerce")
    # Объединяем в один DataFrame и удаляем строки с пропусками
    df = pd.DataFrame({"y": yv, "x": xv}).dropna()
    # Проверяем, что после очистки осталось хотя бы 20 наблюдений (иначе тест ненадёжен)
    if len(df) < 20:
        raise ValueError("Слишком мало наблюдений для теста Энгла–Грейнджера (менее 20)")
    # Добавляем столбец константы для свободного члена регрессии
    X = sm.add_constant(df["x"])
    # Оцениваем долгосрочное уравнение y = alpha + beta * x методом наименьших квадратов
    model = sm.OLS(df["y"], X).fit()
    # Получаем остатки регрессии – предполагаемые отклонения от долгосрочного равновесия
    resid = model.resid
    # Выполняем ADF-тест для остатков (без константы и тренда, т.к. среднее остатков ≈ 0)
    adf = adfuller(resid, autolag="AIC", regression='n')
    # Извлекаем p‑значение (второй элемент кортежа, возвращаемого adfuller)
    p = float(adf[1])
    # Коинтеграция считается присутствующей, если p‑значение меньше 0.05
    coint = p < 0.05
    # Извлекаем коэффициенты долгосрочного уравнения (с защитой от отсутствия)
    beta0 = float(model.params.get("const", np.nan))
    beta1 = float(model.params.get("x", np.nan))
    # Возвращаем структурированный результат в виде dataclass
    return EngleGrangerResult(
        y=y_name,
        x=x_name,
        adf_p_resid=p,
        cointegrated=coint,
        beta0=beta0,
        beta1=beta1,
    )

