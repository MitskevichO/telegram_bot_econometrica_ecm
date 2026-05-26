from __future__ import annotations

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton


def main_menu_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="📊 Загрузить данные", callback_data="menu:upload")],
            [InlineKeyboardButton(text="ℹ️ Как использовать", callback_data="menu:howto")],
        ]
    )


def after_upload_kb() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="🔍 Проверить стационарность", callback_data="act:stationarity")],
            [InlineKeyboardButton(text="📈 Построить графики", callback_data="act:plots")],
            [InlineKeyboardButton(text="📋 Показать данные", callback_data="act:stats")],
            [InlineKeyboardButton(text="🔄 Начать заново", callback_data="act:reset")],
        ]
    )


def reset_reply_kb() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="/start")],
            [KeyboardButton(text="🔄 Начать сначала")],
        ],
        resize_keyboard=True,
        one_time_keyboard=False,
    )