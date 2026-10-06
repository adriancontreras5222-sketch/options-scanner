"""Settings loaded from the .env file next to app.py."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
load_dotenv(ROOT / ".env")

DEFAULT_WATCHLIST = "SPY QQQ IWM AAPL MSFT NVDA AMZN META GOOGL AMD TSLA JPM XOM"


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


@dataclass
class Settings:
    apify_token: str = field(default_factory=lambda: os.getenv("APIFY_TOKEN", "").strip())
    massive_key: str = field(default_factory=lambda: os.getenv("MASSIVE_API_KEY", "").strip())
    chain_provider: str = field(default_factory=lambda: os.getenv("CHAIN_PROVIDER", "apify").strip().lower())
    apify_actor: str = field(
        default_factory=lambda: os.getenv("APIFY_ACTOR", "ahmed_jasarevic/yahoo-finance-options").strip()
    )
    apify_max_usd_per_scan: float = field(default_factory=lambda: _float("APIFY_MAX_USD_PER_SCAN", 1.50))
    risk_free_rate: float = field(default_factory=lambda: _float("RISK_FREE_RATE", 0.04))
    watchlist: str = field(default_factory=lambda: os.getenv("WATCHLIST", DEFAULT_WATCHLIST))


SETTINGS = Settings()

# Apify price per contract row for the chosen actor (free tier). Used for cost estimates only.
APIFY_USD_PER_ROW = 0.0012
APIFY_FREE_ROW_CAP = 50
