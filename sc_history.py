"""Daily price history (Massive first, Yahoo fallback) and past earnings dates. Cached on disk per day."""
from __future__ import annotations

import datetime as dt
import json
import time

import pandas as pd
import requests

from sc_config import DATA_DIR, SETTINGS

CACHE = DATA_DIR / "cache"
CACHE.mkdir(exist_ok=True)
_MASSIVE_HOSTS = ["https://api.massive.com", "https://api.polygon.io"]
_last_call = [0.0]
MASSIVE_MIN_GAP_S = 12.5  # free plan allows 5 calls per minute


def _cache_path(kind: str, ticker: str):
    return CACHE / f"{kind}_{ticker}_{dt.date.today().isoformat()}.csv"


def _massive_bars(ticker: str, days: int) -> pd.DataFrame:
    end = dt.date.today()
    start = end - dt.timedelta(days=int(days * 1.5))
    wait = MASSIVE_MIN_GAP_S - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    last_err = None
    for host in _MASSIVE_HOSTS:
        url = f"{host}/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
        try:
            r = requests.get(url, params={"adjusted": "true", "sort": "asc", "limit": 5000,
                                          "apiKey": SETTINGS.massive_key}, timeout=30)
            _last_call[0] = time.time()
            if r.status_code == 429:
                time.sleep(MASSIVE_MIN_GAP_S)
                continue
            r.raise_for_status()
            res = r.json().get("results") or []
            if not res:
                return pd.DataFrame()
            df = pd.DataFrame(res).rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
            df["date"] = pd.to_datetime(df["t"], unit="ms").dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.date
            return df[["date", "open", "high", "low", "close", "volume"]]
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"Massive failed for {ticker}: {last_err}")


def _yahoo_bars(ticker: str, days: int) -> pd.DataFrame:
    import yfinance as yf

    h = yf.Ticker(ticker).history(period="2y" if days > 260 else "1y", auto_adjust=True)
    if h.empty:
        return pd.DataFrame()
    h = h.reset_index()
    h = h.rename(columns=str.lower)
    h["date"] = pd.to_datetime(h["date"]).dt.date
    return h[["date", "open", "high", "low", "close", "volume"]]


def _alpaca_bars(ticker: str, days: int, key: str, secret: str) -> pd.DataFrame:
    start = (dt.date.today() - dt.timedelta(days=int(days * 1.5))).isoformat()
    h = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    r = requests.get(f"https://data.alpaca.markets/v2/stocks/{ticker}/bars", headers=h, timeout=30,
                     params={"timeframe": "1Day", "start": start, "limit": 10000, "adjustment": "all", "feed": "iex"})
    r.raise_for_status()
    bars = r.json().get("bars") or []
    if not bars:
        return pd.DataFrame()
    df = pd.DataFrame(bars).rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df["date"] = pd.to_datetime(df["t"]).dt.tz_convert("America/New_York").dt.date
    return df[["date", "open", "high", "low", "close", "volume"]]


def get_bars(ticker: str, days: int = 400, use_massive: bool = True, alpaca: tuple | None = None) -> tuple[pd.DataFrame, str]:
    """Returns (bars, source). Bars are daily, oldest first."""
    p = _cache_path("bars", ticker)
    if p.exists():
        df = pd.read_csv(p, parse_dates=["date"])
        df["date"] = df["date"].dt.date
        return df, "cache"
    df, src = pd.DataFrame(), ""
    if alpaca:
        try:
            df, src = _alpaca_bars(ticker, days, *alpaca), "Alpaca"
        except Exception:
            df = pd.DataFrame()
    if df.empty and use_massive and SETTINGS.massive_key:
        try:
            df, src = _massive_bars(ticker, days), "Massive"
        except Exception:
            df = pd.DataFrame()
    if df.empty:
        try:
            df, src = _yahoo_bars(ticker, days), "Yahoo"
        except Exception:
            df = pd.DataFrame()
    if not df.empty:
        df.to_csv(p, index=False)
    return df, src


def past_earnings(ticker: str, n: int = 12) -> list[dict]:
    """Past earnings as [{'date': date, 'timing': 'BMO'|'AMC'}], newest first. Source: Yahoo."""
    p = CACHE / f"earn_{ticker}_{dt.date.today().isoformat()}.json"
    if p.exists():
        raw = json.loads(p.read_text())
        return [{"date": dt.date.fromisoformat(x["date"]), "timing": x["timing"]} for x in raw]
    out = []
    try:
        import yfinance as yf

        ed = yf.Ticker(ticker).get_earnings_dates(limit=n + 8)
        if ed is not None and not ed.empty:
            now = pd.Timestamp.now(tz=ed.index.tz)
            for ts in ed.index:
                if ts >= now:
                    continue
                local = ts.tz_convert("America/New_York") if ts.tzinfo else ts
                timing = "AMC" if local.hour >= 12 else "BMO"
                out.append({"date": local.date(), "timing": timing})
    except Exception:
        out = []
    out = out[:n]
    p.write_text(json.dumps([{"date": x["date"].isoformat(), "timing": x["timing"]} for x in out]))
    return out
