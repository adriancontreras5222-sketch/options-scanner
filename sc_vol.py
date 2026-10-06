"""Realized volatility, trend signals, earnings move history, and a local IV history store."""
from __future__ import annotations

import datetime as dt
import math
import sqlite3

import numpy as np
import pandas as pd

from sc_config import DATA_DIR

IV_DB = DATA_DIR / "iv_history.sqlite"


def realized_vol(close: pd.Series, window: int) -> float:
    r = np.log(close / close.shift(1)).dropna().tail(window)
    if len(r) < max(5, window // 2):
        return float("nan")
    return float(r.std(ddof=1) * math.sqrt(252))


def rsi(close: pd.Series, n: int = 14) -> float:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return float((100 - 100 / (1 + rs)).iloc[-1])


def stock_profile(bars: pd.DataFrame) -> dict:
    """Volatility and trend facts used by every scanner."""
    if bars is None or len(bars) < 60:
        return {}
    c = bars["close"].astype(float).reset_index(drop=True)
    sma50 = c.rolling(50).mean().iloc[-1]
    sma200 = c.rolling(200).mean().iloc[-1] if len(c) >= 200 else float("nan")
    last = float(c.iloc[-1])
    ret20 = float(c.iloc[-1] / c.iloc[-21] - 1) if len(c) > 21 else float("nan")
    r = rsi(c)
    hv20, hv60 = realized_vol(c, 20), realized_vol(c, 60)
    if not math.isnan(sma200):
        if last > sma50 > sma200 and 50 <= r <= 72 and ret20 > 0:
            trend = "bullish"
        elif last < sma50 < sma200 and 28 <= r <= 50 and ret20 < 0:
            trend = "bearish"
        else:
            trend = "neutral"
    else:
        trend = "neutral"
    return {
        "last": last, "sma50": float(sma50), "sma200": float(sma200), "rsi": r, "ret20": ret20,
        "hv20": hv20, "hv60": hv60, "hv_forecast": float(np.nanmean([hv20, hv60])), "trend": trend,
    }


def earnings_moves(bars: pd.DataFrame, events: list[dict]) -> list[dict]:
    """Close-to-close reaction for each past report. AMC: day after vs report day. BMO: report day vs prior day."""
    if bars is None or bars.empty:
        return []
    b = bars.set_index("date")["close"].astype(float)
    dates = list(b.index)
    out = []
    for ev in events:
        d = ev["date"]
        later = [x for x in dates if x >= d]
        if not later:
            continue
        i = dates.index(later[0])
        if ev["timing"] == "AMC":
            if later[0] != d or i + 1 >= len(dates):
                continue
            before, after = b.iloc[i], b.iloc[i + 1]
        else:
            if i == 0:
                continue
            before, after = b.iloc[i - 1], b.iloc[i]
        out.append({"date": d, "timing": ev["timing"], "move": float(after / before - 1)})
    return out


# ---------------- IV history (builds IV rank over time) ----------------

def _db():
    con = sqlite3.connect(IV_DB)
    con.execute("CREATE TABLE IF NOT EXISTS iv (ticker TEXT, day TEXT, atm_iv REAL, PRIMARY KEY (ticker, day))")
    return con


def atm_iv(chain_t: pd.DataFrame, target_dte: int = 30) -> float:
    """Average call/put IV at the strike nearest spot, on the expiry closest to target_dte."""
    if chain_t.empty:
        return float("nan")
    exps = chain_t.drop_duplicates("expiration")[["expiration", "dte"]]
    exp = exps.iloc[(exps.dte - target_dte).abs().argsort().iloc[0]]["expiration"]
    sub = chain_t[(chain_t.expiration == exp) & chain_t.iv.notna()]
    if sub.empty:
        return float("nan")
    spot = sub.spot.iloc[0]
    k = sub.iloc[(sub.strike - spot).abs().argsort().iloc[0]]["strike"]
    return float(sub[sub.strike == k].iv.mean())


def record_iv(ticker: str, value: float) -> None:
    if value is None or math.isnan(value):
        return
    with _db() as con:
        con.execute("INSERT OR REPLACE INTO iv VALUES (?, ?, ?)", (ticker, dt.date.today().isoformat(), value))


def iv_rank(ticker: str, current: float, min_days: int = 20) -> tuple[float, int]:
    """(rank 0-100 or nan, days of history). Uses up to one year of your own saved scans."""
    with _db() as con:
        rows = con.execute(
            "SELECT atm_iv FROM iv WHERE ticker=? AND day>=?",
            (ticker, (dt.date.today() - dt.timedelta(days=365)).isoformat()),
        ).fetchall()
    vals = [r[0] for r in rows]
    if len(vals) < min_days or math.isnan(current):
        return float("nan"), len(vals)
    lo, hi = min(vals), max(vals)
    return (float("nan") if hi == lo else 100 * (current - lo) / (hi - lo)), len(vals)
