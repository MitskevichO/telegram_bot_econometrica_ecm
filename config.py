from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    token: str
    data_folder: str = "data"
    report_folder: str = "reports"

    # Session TTL in seconds (30 minutes)
    session_ttl_s: int = int(os.getenv("SESSION_TTL_S", "1800"))

    # Visualization
    locale: str = os.getenv("BOT_LOCALE", "ru_RU")


def load_settings() -> Settings:
    token = os.getenv("BOT_TOKEN", "ВСтавьте токен своего бота!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!").strip()
    if not token:
        raise RuntimeError(
            "BOT_TOKEN is not set. Create a .env file with BOT_TOKEN=... or set it in environment variables."
        )
    return Settings(token=token)
