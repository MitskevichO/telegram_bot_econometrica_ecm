from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd
from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message, BufferedInputFile

from econobot.keyboards.analysis_menu import (
    analysis_menu_kb,
    ci_level_kb,
    forecasting_menu_kb,
    lags_dy_kb,
)
from econobot.keyboards.main_menu import after_upload_kb, main_menu_kb, reset_reply_kb
from econobot.keyboards.roles import roles_kb
from econobot.keyboards.tests_menu import diagnostics_menu_kb
from econobot.src.cointegration import engle_granger_test
from econobot.src.data_access import pick_yx_columns, session_df
from econobot.src.diagnostics import (
    breusch_godfrey_lm,
    breusch_godfrey_resid_exog,
    chow_test_simple,
    white_test,
)
from econobot.src.forecast import (
    forecast_with_ci_ols_with_x,
    forecast_ecm,
)
from econobot.src.interpreter import (
    format_validation_summary,
    interpret_ecm_result,
)
from econobot.src.models import build_ecm
from econobot.src.reports import (
    build_full_excel_report_bytes,
    build_sample_file_bytes,
)
from econobot.src.stationarity import stationarity_adf_kpss
from econobot.src.validators import load_table_from_bytes, validate_table
from econobot.src.visualizer import (
    plot_acf_pacf_panels,
    plot_corr_heatmap,
    plot_residuals_series,
    plot_stationarity_panels,
    plot_timeseries,
)
from econobot.states.states import Flow
from econobot.utils.cache import InMemorySessionStore
from econobot.utils.file_handler import save_user_upload

common_router = Router(name="common")
log = logging.getLogger(__name__)

session_store = InMemorySessionStore(ttl_s=3600)


def _add_history(session: dict, entry_type: str, title: str, content: str) -> None:
    history = session.get("history", [])
    history.append({
        "type": entry_type,
        "title": title,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })
    session["history"] = history


# ---------------------- Старт и общее меню ----------------------
@common_router.message(CommandStart())
async def start_cmd(message: Message, state: FSMContext) -> None:
    await state.clear()
    user_id = message.from_user.id
    await session_store.delete_session(user_id)
    await message.answer(
        "👋 Привет! Я бот для эконометрического анализа временных рядов.\n"
        "Я умею:\n"
        "✅ Проверять стационарность и коинтеграцию\n"
        "✅ Проводить диагностику остатков\n"
        "✅ Строить прогнозы с доверительными интервалами\n\n"
        "📎 Отправьте Excel файл с двумя временными рядами.",
        reply_markup=main_menu_kb(),
    )
    await message.answer(
        "Сбросить сессию можно в любой момент кнопкой ниже:",
        reply_markup=reset_reply_kb(),
    )


@common_router.message(F.text == "🔄 Начать сначала")
async def reset_by_reply(message: Message, state: FSMContext) -> None:
    await message.answer("Сброс сессии...")
    await start_cmd(message, state)


@common_router.callback_query(F.data == "menu:howto")
async def howto(call: CallbackQuery) -> None:
    await call.answer()
    await call.message.answer(
        "Как это работает:\n"
        "- Загрузите .xlsx (колонка с датами название колонки(date) + 2 числовые колонки, например назовите их : Х и Y)\n"
        "- Затем вы можете запустить тесты стационарности, коинтеграции и построить прогноз\n\n"
        "Прогнозирование: вы вводите будущие значения X, бот возвращает Ŷ.\n"
        "Совет: называйте колонки: дата, Y, X или подобным образом.",
        reply_markup=main_menu_kb(),
    )


@common_router.callback_query(F.data == "menu:sample")
async def sample(call: CallbackQuery) -> None:
    await call.answer()
    content = build_sample_file_bytes()
    await call.message.answer_document(
        document=BufferedInputFile(content, filename="sample_timeseries.xlsx"),
        caption="Пример файла (.xlsx).",
    )


@common_router.callback_query(F.data == "menu:upload")
async def upload_hint(call: CallbackQuery, state: FSMContext) -> None:
    await call.answer()
    await state.set_state(Flow.waiting_file)
    await call.message.answer("Отправьте ваш .xlsx  файл.")


# ---------------------- Загрузка файла ----------------------
@common_router.message(Flow.waiting_file, F.document)
async def handle_file(message: Message, state: FSMContext) -> None:
    assert message.document is not None
    user_id = message.from_user.id if message.from_user else 0

    try:
        file_bytes, original_name, _ = await save_user_upload(message, state)
        df = load_table_from_bytes(file_bytes=file_bytes, filename=original_name)
        v = validate_table(df)

        await session_store.set_session(
            user_id=user_id,
            payload={
                "file_path": None,
                "filename": original_name,
                "df_json": df.to_json(orient="split", date_format="iso"),
                "validation": v.model_dump(),
                "ci_level": 95,
                "roles": {"y": None, "x": None},
                "meta": {},
                "lags_dy": 1,
                "history": [],
            },
        )
        await state.set_state(Flow.choosing_roles)

        await message.answer(
            format_validation_summary(v),

        )
        await message.answer(
            "Выберите роли для ваших рядов:\n"
            "- Зависимая переменная → Y\n"
            "- Независимая переменная → X\n"
            "Затем нажмите 'Далее →'",
            reply_markup=roles_kb(v.numeric_cols, None, None),
        )
    except Exception as e:
        log.exception("Upload processing failed: %s", e)
        await message.answer(
            "❌ Не удалось обработать файл.\n"
            "Убедитесь, что это .xlsx или .csv с колонкой дат и минимум двумя числовыми колонками.\n"
            "Попробуйте снова или скачайте пример.",
            reply_markup=main_menu_kb(),
        )


@common_router.callback_query(F.data == "act:reset")
async def reset(call: CallbackQuery, state: FSMContext) -> None:
    await call.answer()
    user_id = call.from_user.id
    await session_store.delete_session(user_id)
    await state.clear()
    await call.message.answer("Сессия очищена. Загрузите новый файл.", reply_markup=main_menu_kb())
    await call.message.answer("Сбросить сессию можно в любой момент:", reply_markup=reset_reply_kb())


# ---------------------- Выбор ролей Y/X ----------------------
@common_router.callback_query(Flow.choosing_roles, F.data.startswith("role:"))
async def choose_roles(call: CallbackQuery, state: FSMContext) -> None:
    await call.answer()
    user_id = call.from_user.id
    session = await session_store.get_session(user_id)
    if not session:
        await call.message.answer("Нет активной сессии. Загрузите данные.", reply_markup=main_menu_kb())
        return

    v = session["validation"]
    roles = session.get("roles") or {"y": None, "x": None}

    if call.data == "role:cancel":
        await state.set_state(Flow.ready)
        await call.message.answer("Выбор ролей пропущен (будет использовано авто-определение).", reply_markup=after_upload_kb())
        return

    if call.data == "role:done":
        if not roles.get("y") or not roles.get("x") or roles["y"] == roles["x"]:
            await call.message.answer("❌ Пожалуйста, выберите разные колонки для Y и X, затем нажмите 'Далее →'.")
            await call.message.edit_reply_markup(
                reply_markup=roles_kb(v["numeric_cols"], roles.get("y"), roles.get("x"))
            )
            return
        session["roles"] = roles
        await session_store.set_session(user_id, session)
        await state.set_state(Flow.entering_meta_y)
        await call.message.answer(
            "Введите название показателя Y и единицы измерения в одной строке, например: `Основные фонды, тыс.BYN`.",
            parse_mode="Markdown",
        )
        return

    _, side, col = call.data.split(":", 2)
    if col not in v["numeric_cols"]:
        await call.message.answer("Неизвестная колонка.")
        return
    if roles.get(side) == col:
        await call.answer("Уже выбрано.")
        return
    roles[side] = col
    session["roles"] = roles
    await session_store.set_session(user_id, session)
    await call.message.edit_reply_markup(
        reply_markup=roles_kb(v["numeric_cols"], roles.get("y"), roles.get("x"))
    )


# ---------------------- Описательная информация ----------------------
@common_router.message(Flow.entering_meta_y)
async def meta_y(message: Message, state: FSMContext) -> None:
    user_id = message.from_user.id
    session = await session_store.get_session(user_id)
    if not session:
        await message.answer("Нет активной сессии. Загрузите данные.", reply_markup=main_menu_kb())
        return
    session.setdefault("meta", {})["y"] = message.text.strip()
    await session_store.set_session(user_id, session)
    await state.set_state(Flow.entering_meta_x)
    await message.answer(
        "Введите название показателя X и единицы измерения в одной строке, например: `Капиталовложения, BYN`.",
        parse_mode="Markdown",
    )


@common_router.message(Flow.entering_meta_x)
async def meta_x(message: Message, state: FSMContext) -> None:
    user_id = message.from_user.id
    session = await session_store.get_session(user_id)
    if not session:
        await message.answer("Нет активной сессии. Загрузите данные.", reply_markup=main_menu_kb())
        return
    session.setdefault("meta", {})["x"] = message.text.strip()
    await session_store.set_session(user_id, session)
    await state.set_state(Flow.ready)
    await message.answer("✅ Описательная информация сохранена. Теперь можно приступать к анализу.", reply_markup=after_upload_kb())


# ---------------------- Базовые действия ----------------------
@common_router.callback_query(F.data == "act:stats")
async def show_data(call: CallbackQuery, state: FSMContext) -> None:
    await call.answer()
    user_id = call.from_user.id
    session = await session_store.get_session(user_id)
    if not session:
        await call.message.answer("Нет активной сессии. Загрузите данные.", reply_markup=main_menu_kb())
        return

    df = session_df(session)
    head = df.head(15).to_string()
    n_rows, n_cols = df.shape
    await call.message.answer(
        f"📄 Предпросмотр данных (первые 15 строк из {n_rows} строк, {n_cols} колонок):\n\n<pre>{head}</pre>",
        parse_mode="HTML",
    )
    await call.message.answer("Далее:", reply_markup=analysis_menu_kb())


@common_router.callback_query(F.data == "act:plots")
async def plots_entry(call: CallbackQuery, state: FSMContext) -> None:
    await call.answer()
    user_id = call.from_user.id
    session = await session_store.get_session(user_id)
    if not session:
        await call.message.answer("Нет активной сессии. Загрузите данные.", reply_markup=main_menu_kb())
        return
    df = session_df(session)
    v = session["validation"]
    y_col, x_col = pick_yx_columns(session)
    date_col = v["date_col"]

    png_ts = plot_timeseries(df, date_col=date_col, y_col=y_col, x_col=x_col)
    await call.message.answer_photo(
        photo=BufferedInputFile(png_ts, filename="timeseries.png"),
        caption="📈 Временные ряды (Y и X)",
    )

    png_acf_y = plot_acf_pacf_panels(df[y_col], title=f"Автокорреляция (ACF) и частная автокорреляция (PACF):  {y_col}")
    await call.message.answer_photo(
        photo=BufferedInputFile(png_acf_y, filename="acf_pacf_y.png"),
        caption=f"Автокорреляция (ACF) и частная автокорреляция (PACF):  {y_col}",
    )
    png_acf_x = plot_acf_pacf_panels(df[x_col], title=f"Автокорреляция (ACF) и частная автокорреляция (PACF):  {x_col}")
    await call.message.answer_photo(
        photo=BufferedInputFile(png_acf_x, filename="acf_pacf_x.png"),
        caption=f"Автокорреляция (ACF) и частная автокорреляция (PACF):  {x_col}",
    )

    # Блок OLS-остатков удалён по просьбе пользователя

    _add_history(session, "plots", "Построены графики", "Временные ряды, ACF/PACF")
    await session_store.set_session(user_id, session)

    await call.message.answer("Далее:", reply_markup=analysis_menu_kb())


@common_router.callback_query(F.data == "act:stationarity")
async def stationarity_entry(call: CallbackQuery, state: FSMContext) -> None:
    await call.answer()
    user_id = call.from_user.id
    session = await session_store.get_session(user_id)
    if not session:
        await call.message.answer("Нет активной сессии. Загрузите данные.", reply_markup=main_menu_kb())
        return

    df = session_df(session)
    v = session["validation"]
    y_col, x_col = pick_yx_columns(session)
    date_col = v["date_col"]

    try:
        y_res = stationarity_adf_kpss(df[y_col], name=y_col)
        x_res = stationarity_adf_kpss(df[x_col], name=x_col)

        explanation = (
            "📖 **Пояснение к тесту ADF:**\n"
            "Нулевая гипотеза H₀: ряд имеет единичный корень (нестационарен).\n"
            "Если p-value < 0.05, мы отвергаем H₀ и считаем ряд стационарным.\n\n"
        )

        text = (
            f"**СТАЦИОНАРНОСТЬ (тест ADF)**\n"
            f"📊 Ряд {y_col}: p-value = {y_res.adf_p:.4f} → I({y_res.integration_order})\n"
            f"📊 Ряд {x_col}: p-value = {x_res.adf_p:.4f} → I({x_res.integration_order})\n\n"
            f"{explanation}"
        )
        await call.message.answer(text, parse_mode="Markdown")

        png_y = plot_stationarity_panels(df, date_col=date_col, series_col=y_col)
        await call.message.answer_photo(
            photo=BufferedInputFile(png_y, filename=f"stationarity_{y_col}.png"),
            caption=f"Панели стационарности: {y_col}",
        )
        png_x = plot_stationarity_panels(df, date_col=date_col, series_col=x_col)
        await call.message.answer_photo(
            photo=BufferedInputFile(png_x, filename=f"stationarity_{x_col}.png"),
            caption=f"Панели стационарности: {x_col}",
        )

        _add_history(session, "stationarity", "Тесты ADF", text)
        await session_store.set_session(user_id, session)

    except Exception as e:
        log.exception("Stationarity failed: %s", e)
        await call.message.answer("❌ Тесты стационарности не удались на этом наборе данных.")

    await call.message.answer("Далее:", reply_markup=analysis_menu_kb())


# ---------------------- Анализ: коинтеграция ----------------------
@common_router.callback_query(F.data == "an:coint")
async def cointegration(call: CallbackQuery, state: FSMContext) -> None:
    await call.answer()
    user_id = call.from_user.id
    session = await session_store.get_session(user_id)
    if not session:
        await call.message.answer("Нет активной сессии. Загрузите данные.", reply_markup=main_menu_kb())
        return
    df = session_df(session)
    y_col, x_col = pick_yx_columns(session)
    try:
        res = engle_granger_test(df[y_col], df[x_col], y_name=y_col, x_name=x_col)

        explanation = (
            "📖 **Пояснение к тесту Энгла-Грейнджера:**\n"
            "Нулевая гипотеза H₀: ряды не коинтегрированы (остатки нестационарны).\n"
            "Если p-value (ADF по остаткам) < 0.05, отвергаем H₀ и делаем вывод о коинтеграции.\n\n"
        )

        text = (
            "🔗 **ТЕСТ ЭНГЛА-ГРЕЙНДЖЕРА**\n"
            f"p-value (ADF остатков) = {res.adf_p_resid:.4f}\n"
            f"**ВЫВОД:** {'Ряды коинтегрированы ✅' if res.cointegrated else 'Коинтеграция не обнаружена ❌'}\n\n"
            f"Долгосрочное уравнение: {res.y} = {res.beta0:.4f} + {res.beta1:.4f}·{res.x} + ε\n\n"
            f"{explanation}"
        )
        await call.message.answer(text, parse_mode="Markdown", reply_markup=analysis_menu_kb())
        session["cointegration_result"] = text
        _add_history(session, "cointegration", "Тест Энгла-Грейнджера", text)
        await session_store.set_session(user_id, session)
    except Exception as e:
        log.exception("Cointegration failed: %s", e)
        await call.message.answer("❌ Тест коинтеграции не удался на этом наборе данных.", reply_markup=analysis_menu_kb())


# ---------------------- Настройка ECM ----------------------
@common_router.callback_query(F.data == "an:ecm_settings")
async def ecm_settings(call: CallbackQuery, state: FSMContext) -> None:
    await call.answer()
    user_id = call.from_user.id
    session = await session_store.get_session(user_id)
    if not session:
        await call.message.answer("Нет активной сессии. Загрузите данные.", reply_markup=main_menu_kb())
        return
    current = session.get("lags_dy", 1)
    await state.set_state(Flow.selecting_lags_dy)
    await call.message.answer(
        "Выберите количество лагов ΔY в ECM модели:",
        reply_markup=lags_dy_kb(current),
    )


@common_router.callback_query(Flow.selecting_lags_dy, F.data.startswith("lags:"))
async def set_lags_dy(call: CallbackQuery, state: FSMContext) -> None:
    await call.answer()
    user_id = call.from_user.id
    action = call.data.split(":")[1]

    if action == "back":
        await state.set_state(Flow.ready)
        await call.message.answer("Настройки ECM закрыты.", reply_markup=analysis_menu_kb())
        return

    lags = int(action)
    session = await session_store.get_session(user_id)
    if session:
        session["lags_dy"] = lags
        await session_store.set_session(user_id, session)
    await state.set_state(Flow.ready)
    await call.message.answer(
        f"✅ Количество лагов ΔY установлено: {lags}. Теперь можно строить ECM модель.",
        reply_markup=analysis_menu_kb(),
    )


# ---------------------- Построение ECM ----------------------
@common_router.callback_query(F.data == "an:ecm")
async def ecm(call: CallbackQuery, state: FSMContext) -> None:
    await call.answer()
    user_id = call.from_user.id
    session = await session_store.get_session(user_id)
    if not session:
        await call.message.answer("Нет активной сессии. Загрузите данные.", reply_markup=main_menu_kb())
        return
    df = session_df(session)
    y_col, x_col = pick_yx_columns(session)

    lags_dy = session.get("lags_dy", 1)
    try:
        model_result, ecm_fitted = build_ecm(df, y_col=y_col, x_col=x_col, lags_dy=lags_dy)
        # Сохраняем остатки и экзогенные переменные ECM в сессию для диагностики
        session["ecm_resid"] = ecm_fitted.resid.tolist()
        session["ecm_exog"] = ecm_fitted.model.exog.tolist()
        session["ecm_exog_names"] = list(ecm_fitted.model.exog_names)
        session["ecm_has"] = True

        # Генерируем и отправляем график остатков ECM модели
        png_res_ecm = plot_residuals_series(ecm_fitted.resid, title="Остатки ECM модели")
        await call.message.answer_photo(
            photo=BufferedInputFile(png_res_ecm, filename="ecm_residuals.png"),
            caption="📉 Временной ряд остатков модели коррекции ошибок (ECM)",
        )

        meta = session.get("meta", {})
        y_meta = meta.get("y", "")
        x_meta = meta.get("x", "")
        text = interpret_ecm_result(model_result, y_meta, x_meta, y_col, x_col)
        await call.message.answer(text, parse_mode="Markdown", reply_markup=analysis_menu_kb())
        session["ecm_result"] = text
        _add_history(session, "ecm", "Модель коррекции ошибок", text)
        await session_store.set_session(user_id, session)
    except Exception as e:
        log.exception("ECM failed: %s", e)
        await call.message.answer("❌ Построение ECM не удалось на этом наборе данных.", reply_markup=analysis_menu_kb())


# ---------------------- Диагностика ----------------------
@common_router.callback_query(F.data == "an:diag")
async def diagnostics_entry(call: CallbackQuery) -> None:
    await call.answer()
    await call.message.answer("Выберите диагностические тесты:", reply_markup=diagnostics_menu_kb())


@common_router.callback_query(F.data.in_({'t:white', 't:bg', 't:all', 't:back'}))
async def diagnostics_tests(call: CallbackQuery, state: FSMContext) -> None:
    await call.answer()
    if call.data == "t:back":
        await call.message.answer("Назад.", reply_markup=analysis_menu_kb())
        return

    user_id = call.from_user.id
    session = await session_store.get_session(user_id)
    if not session:
        await call.message.answer("Нет активной сессии. Загрузите данные.", reply_markup=main_menu_kb())
        return

    # Проверяем, есть ли построенная ECM модель
    if not session.get("ecm_has"):
        await call.message.answer(
            "❌ Сначала постройте ECM модель (кнопка 'Построить ECM модель'), а затем запускайте диагностику остатков.",
            reply_markup=analysis_menu_kb()
        )
        return

    # Восстанавливаем остатки и экзогенные переменные ECM
    resid = pd.Series(session["ecm_resid"])
    exog = np.array(session["ecm_exog"])
    if exog.shape[1] > 1:
        X_for_white = pd.DataFrame(exog[:, 1:], columns=session.get("ecm_exog_names", [None])[1:])
    else:
        X_for_white = pd.DataFrame()

    results = []
    explanations = []

    try:
        if call.data in {"t:white", "t:all"}:
            if X_for_white.empty or X_for_white.shape[1] == 0:
                await call.message.answer("⚠️ Тест Уайта невозможен: в модели только константа.")
            else:
                res_white = white_test(resid, exog=X_for_white)
                results.append(res_white)
                explanations.append(
                    "📖 **Тест Уайта (White)** проверяет **гетероскедастичность** – меняется ли разброс остатков во времени.\n"
                    "   • H₀: дисперсия остатков постоянна (гомоскедастичность).\n"
                    "   • Если p < 0.05 – гетероскедастичность есть, стандартные ошибки ненадёжны.\n"
                )
        if call.data in {"t:bg", "t:all"}:
            exog_df = pd.DataFrame(exog, columns=session.get("ecm_exog_names"))
            res_bg = breusch_godfrey_resid_exog(resid, exog=exog_df, nlags=1)
            results.append(res_bg)
            explanations.append(
                "📖 **Тест Бройша-Годфри (Breusch-Godfrey)** проверяет **автокорреляцию остатков**.\n"
                "   • H₀: нет автокорреляции порядка до 1.\n"
                "   • Если p < 0.05 – автокорреляция есть, модель упустила динамику.\n"
            )
    except Exception as e:
        log.exception("Diagnostics failed: %s", e)
        await call.message.answer("❌ Диагностика не удалась на этом наборе данных.", reply_markup=diagnostics_menu_kb())
        return

    lines = ["📉 **ДИАГНОСТИКА ОСТАТКОВ ECM**\n"]
    for r in results:
        mark = "✅" if r.ok else "❌"
        lines.append(f"{mark} **{r.name}**: p-value = {r.p_value:.4f}")
        if not r.ok:
            if r.name == "White":
                lines.append("   ⚠️ Обнаружена гетероскедастичность (p < 0.05). Используйте робастные стандартные ошибки.")
            elif r.name == "Breusch-Godfrey":
                lines.append("   ⚠️ Обнаружена автокорреляция остатков (p < 0.05). Добавьте лаги или измените спецификацию.")
        lines.append("")
    output = "\n".join(lines)
    for expl in explanations:
        output += "\n" + expl
    await call.message.answer(output, parse_mode="Markdown", reply_markup=diagnostics_menu_kb())

    session["diagnostics_result"] = output
    _add_history(session, "diagnostics", "Диагностика остатков ECM", output)
    await session_store.set_session(user_id, session)


# ---------------------- Тест Чоу ----------------------
@common_router.callback_query(F.data == "t:chow")
async def chow_ask_year(call: CallbackQuery, state: FSMContext) -> None:
    await call.answer()
    user_id = call.from_user.id
    session = await session_store.get_session(user_id)
    if not session:
        await call.message.answer("Нет активной сессии. Загрузите данные.", reply_markup=main_menu_kb())
        return
    await state.set_state(Flow.waiting_chow_year)
    await call.message.answer(
        "Введите год, в котором вы хотите проверить структурный разрыв.\n"
        "Например: 2020\n\n"
        "Год должен быть в пределах периода ваших данных.\n\n"
        "📖 **Пояснение:** Нулевая гипотеза теста Чоу – отсутствие структурного разрыва (коэффициенты стабильны).\n"
        "Если p-value < 0.05, структурный разрыв присутствует."
    )


@common_router.message(Flow.waiting_chow_year)
async def process_chow_year(message: Message, state: FSMContext) -> None:
    user_id = message.from_user.id
    session = await session_store.get_session(user_id)
    if not session:
        await message.answer("Нет активной сессии. Загрузите данные.", reply_markup=main_menu_kb())
        return

    try:
        year = int(message.text.strip())
    except ValueError:
        await message.answer("❌ Пожалуйста, введите целое число (год). Например: 2020")
        return

    df = session_df(session)
    v = session["validation"]
    date_col = v["date_col"]

    try:
        numeric_vals = pd.to_numeric(df[date_col], errors='coerce')
        is_year_col = (numeric_vals.notna().all() and
                       (numeric_vals % 1 == 0).all() and
                       1900 <= numeric_vals.min() <= 2100 and
                       1900 <= numeric_vals.max() <= 2100)
    except:
        is_year_col = False

    if is_year_col:
        years = numeric_vals.astype(int)
        min_year, max_year = years.min(), years.max()
        if year < min_year or year > max_year:
            await message.answer(f"❌ Год {year} вне диапазона ({min_year}–{max_year}).")
            return
        break_index = (years <= year).sum()
    else:
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            years = df[date_col].dt.year
            min_year, max_year = years.min(), years.max()
            if year < min_year or year > max_year:
                await message.answer(f"❌ Год {year} вне диапазона ({min_year}–{max_year}).")
                return
            break_index = (years <= year).sum()
        except Exception:
            await message.answer("❌ Не удалось преобразовать даты.")
            return

    if break_index <= 2 or break_index >= len(df) - 2:
        await message.answer("❌ Год слишком близко к началу или концу выборки.")
        return

    y_col, x_col = pick_yx_columns(session)
    d = df[[y_col, x_col]].dropna()

    try:
        result = chow_test_simple(d, y_col=y_col, x_col=x_col, break_index=break_index)
        mark = "✅" if result.ok else "❌"
        output = (
            f"📅 **Тест Чоу (структурный разрыв на {year} год)**\n"
            f"{mark} p-value = {result.p_value:.4f}\n\n"
            f"**Интерпретация:** {'структурный сдвиг присутствует' if not result.ok else 'структурный сдвиг отсутствует'}.\n\n"
            f"*Нулевая гипотеза: отсутствие структурного разрыва (коэффициенты стабильны).*\n"
            f"*Если p < 0.05 → разрыв есть.*"
        )
        await message.answer(output, parse_mode="Markdown")
        _add_history(session, "chow_test", f"Тест Чоу на {year} год", output)
        await session_store.set_session(user_id, session)
    except Exception as e:
        log.exception("Chow test failed: %s", e)
        await message.answer("❌ Не удалось выполнить тест Чоу.")

    await state.set_state(Flow.ready)
    await message.answer("Вернуться к диагностике можно через меню.", reply_markup=diagnostics_menu_kb())


# ---------------------- Прогнозирование ----------------------
@common_router.callback_query(F.data == "an:forecast")
async def forecasting_entry(call: CallbackQuery, state: FSMContext) -> None:
    await call.answer()
    user_id = call.from_user.id
    session = await session_store.get_session(user_id)
    if not session:
        await call.message.answer("Нет активной сессии. Загрузите данные.")
        return
    ci = int(session.get("ci_level", 95))
    ecm_available = True
    await call.message.answer(
        "Опции прогнозирования:",
        reply_markup=forecasting_menu_kb(ci_level=ci, ecm_available=ecm_available)
    )


@common_router.callback_query(F.data == "fc:ci")
async def fc_ci_request(call: CallbackQuery, state: FSMContext) -> None:
    await call.answer()
    user_id = call.from_user.id
    session = await session_store.get_session(user_id)
    if not session:
        await call.message.answer("Нет активной сессии. Загрузите данные.", reply_markup=main_menu_kb())
        return
    y_col, x_col = pick_yx_columns(session)
    await state.set_state(Flow.waiting_x_for_ci)
    await state.update_data(forecast_type="ci")
    ci = int(session.get("ci_level", 95))
    await call.message.answer(
        f"📊 Прогноз с {ci}% доверительным интервалом.\n"
        f"Введите будущие значения для X (`{x_col}`) через пробел или запятую.\n"
        f"Пример: `120 125 130`\n\n"
        f"Я предскажу Y (`{y_col}`) с доверительным интервалом.",
        parse_mode="Markdown",
    )


@common_router.message(Flow.waiting_x_for_ci)
async def process_x_for_ci(message: Message, state: FSMContext) -> None:
    user_id = message.from_user.id
    session = await session_store.get_session(user_id)
    if not session:
        await message.answer("Нет активной сессии. Загрузите данные.", reply_markup=main_menu_kb())
        return

    try:
        parts = message.text.replace(",", " ").split()
        x_future = [float(p) for p in parts]
    except Exception:
        await message.answer("❌ Пожалуйста, введите числа через пробел или запятую.")
        return

    if not x_future:
        await message.answer("❌ Нужно хотя бы одно значение X.")
        return

    df = session_df(session)
    y_col, x_col = pick_yx_columns(session)
    ci = int(session.get("ci_level", 95))
    res = forecast_with_ci_ols_with_x(df, y_col, x_col, x_future, ci)

    output = f"📊 **Прогноз с {ci}% доверительным интервалом** (MAPE = {res.mape:.2f}%)\n\n"
    for xv, yv, lo, hi in zip(res.x_values, res.y_hat, res.lower, res.upper):
        output += f"X = {xv:,.2f}  →  Ŷ = {yv:,.2f}  Доверительный интервал: [{lo:,.2f}, {hi:,.2f}]\n"

    await message.answer(output, parse_mode="Markdown", reply_markup=analysis_menu_kb())
    await state.clear()

    session["forecast_result"] = output
    _add_history(session, "forecast", f"Прогноз с {ci}% Доверительным интервалом(лин.регрессия)", output)
    await session_store.set_session(user_id, session)


@common_router.callback_query(F.data == "fc:set_ci")
async def set_ci(call: CallbackQuery, state: FSMContext) -> None:
    await call.answer()
    await state.set_state(Flow.selecting_ci)
    await call.message.answer("Выберите уровень доверительного интервала:", reply_markup=ci_level_kb())


@common_router.callback_query(F.data == "fc:download")
async def fc_download(call: CallbackQuery, state: FSMContext) -> None:
    await call.answer()
    user_id = call.from_user.id
    session = await session_store.get_session(user_id)
    if not session:
        await call.message.answer("Нет активной сессии. Загрузите данные.", reply_markup=main_menu_kb())
        return

    df = session_df(session)
    v = session["validation"]
    y_col, x_col = pick_yx_columns(session)

    images = {}
    try:
        images["Временные ряды"] = plot_timeseries(df, date_col=v["date_col"], y_col=y_col, x_col=x_col)
        images["ACF/PACF Y"] = plot_acf_pacf_panels(df[y_col], title=f"ACF/PACF: {y_col}")
        images["ACF/PACF X"] = plot_acf_pacf_panels(df[x_col], title=f"ACF/PACF: {x_col}")
        # Если в сессии есть остатки ECM, добавим их график в отчёт
        if session.get("ecm_has") and session.get("ecm_resid"):
            ecm_resid_series = pd.Series(session["ecm_resid"])
            images["Остатки ECM"] = plot_residuals_series(ecm_resid_series, title="Остатки ECM")
        images["Стационарность Y"] = plot_stationarity_panels(df, date_col=v["date_col"], series_col=y_col)
        images["Стационарность X"] = plot_stationarity_panels(df, date_col=v["date_col"], series_col=x_col)
    except Exception as e:
        log.exception("Failed to generate report images: %s", e)

    excel_bytes = build_full_excel_report_bytes(df, session, images)
    await call.message.answer_document(
        document=BufferedInputFile(excel_bytes, filename="econobot_full_report.xlsx"),
        caption="📥 Полный отчёт (Excel)",
    )


@common_router.callback_query(F.data == "fc:ecm")
async def fc_ecm_request(call: CallbackQuery, state: FSMContext) -> None:
    await call.answer()
    user_id = call.from_user.id
    session = await session_store.get_session(user_id)
    if not session:
        await call.message.answer("Нет активной сессии. Загрузите данные.")
        return
    y_col, x_col = pick_yx_columns(session)
    await state.set_state(Flow.waiting_x_for_ecm_forecast)
    await state.update_data(forecast_type="ecm")
    await call.message.answer(
        f"🏭 **ECM прогноз** (с доверительным интервалом).\n"
        f"Введите будущие значения для X (`{x_col}`) через пробел или запятую.\n"
        f"Пример: `120 125 130`\n\n"
        f"Я предскажу Y (`{y_col}`) с помощью модели коррекции ошибок.",
        parse_mode="Markdown"
    )


@common_router.message(Flow.waiting_x_for_ecm_forecast)
async def process_x_for_ecm(message: Message, state: FSMContext) -> None:
    user_id = message.from_user.id
    session = await session_store.get_session(user_id)
    if not session:
        await message.answer("Нет активной сессии. Загрузите данные.")
        return

    try:
        parts = message.text.replace(",", " ").split()
        x_future = [float(p) for p in parts]
    except Exception:
        await message.answer("❌ Пожалуйста, введите числа через пробел или запятую.")
        return

    if not x_future:
        await message.answer("❌ Нужно хотя бы одно значение X.")
        return

    df = session_df(session)
    y_col, x_col = pick_yx_columns(session)
    lags_dy = session.get("lags_dy", 1)
    ci_level = int(session.get("ci_level", 95))

    try:
        res = forecast_ecm(df, y_col, x_col, x_future, lags_dy=lags_dy, ci_level=ci_level)
        output = f"🏭 **ECM прогноз** (MAPE = {res.mape:.2f}%, {ci_level}% Доверительный интервал:)\n\n"
        for xv, yv, lo, hi in zip(res.x_values, res.y_hat, res.lower, res.upper):
            output += f"X = {xv:,.2f}  →  Ŷ = {yv:,.2f}  доверительный интервал:: [{lo:,.2f}, {hi:,.2f}]\n"
        await message.answer(output, parse_mode="Markdown", reply_markup=analysis_menu_kb())
        session["forecast_result"] = output
        _add_history(session, "forecast", "ECM прогноз", output)
        await session_store.set_session(user_id, session)
    except Exception as e:
        log.exception("ECM forecast failed: %s", e)
        await message.answer("❌ ECM прогноз не удался. Убедитесь, что ряды коинтегрированы и данных достаточно.")

    await state.clear()


@common_router.callback_query(Flow.selecting_ci, F.data.startswith("ci:"))
async def set_ci_level(call: CallbackQuery, state: FSMContext) -> None:
    await call.answer()
    user_id = call.from_user.id
    if call.data == "ci:back":
        await state.set_state(Flow.ready)
        await call.message.answer("Назад.", reply_markup=analysis_menu_kb())
        return

    level = int(call.data.split(":")[1])
    session = await session_store.get_session(user_id)
    if session:
        session["ci_level"] = level
        await session_store.set_session(user_id, session)
    await state.set_state(Flow.ready)
    await call.message.answer(f"✅ Уровень доверия установлен на {level}%.", reply_markup=analysis_menu_kb())