from __future__ import annotations

from econobot.src.validators import ValidationResult


def format_validation_summary(v: ValidationResult) -> str:
    cols = ", ".join(v.numeric_cols)

    return (
        "✅ Данные успешно загружены!\n"
        f"📈 Наблюдений: {v.n_obs}\n"
        f"🗓️ Период: {v.period_start} — {v.period_end}\n"
        f"📌 Числовые колонки: {cols}\n\n"

    )


def _sig_stars(p: float) -> str:
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.1:
        return "*"
    return ""


def _extract_unit(display_name: str) -> str:
    """Из строки вида 'ВВП, млрд руб' извлекает 'млрд руб'."""
    if ',' in display_name:
        unit = display_name.split(',', 1)[1].strip()
        if unit:
            return unit
    return "единицу"  # значение по умолчанию


def interpret_ecm_result(model, y_meta: str, x_meta: str, y_col: str, x_col: str) -> str:
    y_display = y_meta if y_meta else y_col
    x_display = x_meta if x_meta else x_col

    # Извлекаем единицы измерения
    y_unit = _extract_unit(y_display)
    x_unit = _extract_unit(x_display)

    const = model.free_coef
    gamma = model.params.get('dx', 0.0)
    alpha = model.alpha
    delta = model.params.get('dy_lag1', 0.0)
    has_delta = 'dy_lag1' in model.params

    p_const = model.pvalues.get('const', 1.0)
    p_gamma = model.pvalues.get('dx', 1.0)
    p_alpha = model.pvalues.get('ect', 1.0)
    p_delta = model.pvalues.get('dy_lag1', 1.0) if has_delta else None

    eq_parts = [f"{const:.2f}"]
    if abs(gamma) > 0:
        eq_parts.append(f"{gamma:.4f}·Δ[{x_display}]")
    if abs(alpha) > 0:
        eq_parts.append(f"{alpha:.4f}·ECT(-1)")
    if has_delta and abs(delta) > 0:
        eq_parts.append(f"{delta:.4f}·Δ[{y_display}](-1)")
    rhs = " + ".join(eq_parts).replace("+ -", "- ")
    equation = f"Δ[{y_display}] = {rhs} + u"

    sig_lines = [
        "**Коэффициенты и их значимость:**",
        f"• Свободный член: {const:.2f}  (p = {p_const:.4f}) {_sig_stars(p_const)}",
        f"• γ (краткосрочный эффект {x_display}): {gamma:.4f}  (p = {p_gamma:.4f}) {_sig_stars(p_gamma)}",
        f"• α (скорость коррекции): {alpha:.4f}  (p = {p_alpha:.4f}) {_sig_stars(p_alpha)}"
    ]
    if has_delta:
        sig_lines.append(f"• δ (инерция {y_display}): {delta:.4f}  (p = {p_delta:.4f}) {_sig_stars(p_delta)}")

    interp_lines = ["**📈 Экономическая интерпретация:**"]
    if p_gamma < 0.1:
        # Используем единицы измерения X и Y
        interp_lines.append(
            f"• Увеличение {x_display} на 1 {x_unit} в текущем периоде приводит к росту "
            f"{y_display} в среднем на {gamma:.2f} {y_unit} в том же периоде (коэффициент значим)."
        )
    else:
        interp_lines.append(f"• Краткосрочный эффект инвестиций статистически незначим (p = {p_gamma:.4f}).")

    if p_alpha < 0.1 and alpha < 0:
        speed = abs(alpha) * 100
        interp_lines.append(
            f"• Коэффициент коррекции α = {alpha:.4f} (p = {p_alpha:.4f}) означает, что за один период (год) "
            f"устраняется {speed:.1f}% отклонения от долгосрочного равновесия. Система возвращается к равновесию."
        )
    elif p_alpha < 0.1 and alpha > 0:
        interp_lines.append(
            f"• Коэффициент коррекции положителен – модель не обеспечивает возврата к равновесию (проверьте спецификацию)."
        )
    else:
        interp_lines.append(
            f"• Коэффициент коррекции статистически незначим (p = {p_alpha:.4f}), механизм коррекции отсутствует."
        )

    if has_delta and p_delta < 0.1:
        interp_lines.append(f"• Инерционный эффект (δ = {delta:.4f}) значим: прошлые изменения {y_display} влияют на текущие.")

    result = [
        "🏗️ **Модель коррекции ошибок (ECM)**",
        "",
        f"**Краткосрочное уравнение:**",
        f"```\n{equation}\n```",
        "",
        f"R² = {model.r2:.4f}",
        "",
        *sig_lines,
        "",
        *interp_lines,
        "",
        "_Примечание: ECT – отклонение от долгосрочного равновесия (остатки коинтеграционного уравнения). Уровни значимости: *** p<0.01, ** p<0.05, * p<0.1_"
    ]
    return "\n".join(result)