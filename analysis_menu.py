from __future__ import annotations

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup


def analysis_menu_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="1)🔗 Тест на коинтеграцию", callback_data="an:coint")],
            [InlineKeyboardButton(text="2)🏗️ Построить ECM модель", callback_data="an:ecm")],
            [InlineKeyboardButton(text="3)⚙️ Настройки ECM (лаги ΔY)", callback_data="an:ecm_settings")],
            [InlineKeyboardButton(text="4)📉 Тестирование последней построенной модели", callback_data="an:diag")],
            [InlineKeyboardButton(text="5)🔮 Прогнозирование", callback_data="an:forecast")],
        ]
    )


def forecasting_menu_kb(ci_level: int, ecm_available: bool = True) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(text="📊 Прогноз с доверительным интервалом(Лин.регрессия)", callback_data="fc:ci")],
        [InlineKeyboardButton(text=f"⚙️ Уровень доверия (сейчас {ci_level}%)", callback_data="fc:set_ci")],
    ]
    if ecm_available:
        rows.append([InlineKeyboardButton(text="🏭 ECM прогноз (с доверительным интервалом)", callback_data="fc:ecm")])
    rows.append([InlineKeyboardButton(text="📥 Скачать результаты", callback_data="fc:download")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def horizon_kb(current: int) -> InlineKeyboardMarkup:
    opts = []
    for h in (1, 3, 6, 12):
        mark = "✅ " if h == current else ""
        opts.append(InlineKeyboardButton(text=f"{mark}{h}", callback_data=f"h:{h}"))
    return InlineKeyboardMarkup(
        inline_keyboard=[
            opts,
            [InlineKeyboardButton(text="⬅️ Назад", callback_data="h:back")],
        ]
    )


def ci_level_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="90%", callback_data="ci:90"),
                InlineKeyboardButton(text="95%", callback_data="ci:95"),
                InlineKeyboardButton(text="99%", callback_data="ci:99"),
            ],
            [InlineKeyboardButton(text="⬅️ Назад", callback_data="ci:back")],
        ]
    )


def lags_dy_kb(current_lags: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text=f"{'✅ ' if current_lags == 0 else ''}0 лагов (без ΔY(-1))", callback_data="lags:0"),
                InlineKeyboardButton(text=f"{'✅ ' if current_lags == 1 else ''}1 лаг (с ΔY(-1))", callback_data="lags:1"),
            ],
            [InlineKeyboardButton(text="⬅️ Назад", callback_data="lags:back")],
        ]
    )