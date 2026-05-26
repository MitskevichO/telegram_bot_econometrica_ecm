from __future__ import annotations

from typing import Tuple

from aiogram.fsm.context import FSMContext
from aiogram.types import Message

async def save_user_upload(message: Message, state: FSMContext) -> Tuple[bytes, str, None]:
    """
    Downloads the uploaded file and returns raw bytes.

    This simplified version does not write uploads to disk.
    """
    bot = message.bot

    # user_id kept for potential future per-user storage; not used in simplified flow.
    doc = message.document
    if doc is None:
        raise ValueError("No document")

    file = await bot.get_file(doc.file_id)
    raw = await bot.download_file(file.file_path)
    data = raw.read()

    filename = (doc.file_name or "upload").replace("/", "_").replace("\\", "_")
    return data, filename, None

