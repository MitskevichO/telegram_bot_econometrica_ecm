from __future__ import annotations

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup


def diagnostics_menu_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Тест Уайта (гетероскедастичность)", callback_data="t:white")],
            [InlineKeyboardButton(text="Тест Бройша-Годфри (автокорреляция)", callback_data="t:bg")],
            [InlineKeyboardButton(text="Тест Чоу (структурные разрывы)", callback_data="t:chow")],
            [InlineKeyboardButton(text="✅ Запустить все тесты", callback_data="t:all")],
            [InlineKeyboardButton(text="⬅️ Назад", callback_data="t:back")],
        ]
    )