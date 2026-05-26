from __future__ import annotations

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup


def roles_kb(numeric_cols: list[str], selected_y: str | None, selected_x: str | None) -> InlineKeyboardMarkup:
    rows: list[list[InlineKeyboardButton]] = []

    for col in numeric_cols:
        y_mark = "✅ " if selected_y == col else ""
        x_mark = "✅ " if selected_x == col else ""
        rows.append(
            [
                InlineKeyboardButton(text=f"{y_mark}Y: {col}", callback_data=f"role:y:{col}"),
                InlineKeyboardButton(text=f"{x_mark}X: {col}", callback_data=f"role:x:{col}"),
            ]
        )

    rows.append([InlineKeyboardButton(text="Далее →", callback_data="role:done")])
    rows.append([InlineKeyboardButton(text="⬅️ Отмена", callback_data="role:cancel")])
    return InlineKeyboardMarkup(inline_keyboard=rows)