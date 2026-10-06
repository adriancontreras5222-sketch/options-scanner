"""Option chain fetching. Two providers return the same normalized DataFrame.

Normalized columns:
  ticker, symbol, type ('call'|'put'), strike, expiration (date), dte, spot,
  bid, ask, mid, last, spread_pct, volume, oi, iv, delta, gamma, theta, vega, earnings_date
"""
from __future__ import annotations

import datetime as dt
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import requests

import sc_bs as bs
from sc_config import APIFY_FREE_ROW_CAP, APIFY_USD_PER_ROW, SETTINGS

COLUMNS = [
    "ticker", "symbol", "type", "strike", "expiration", "dte", "spot", "bid", "ask", "mid", "last",
    "spread_pct", "volume", "oi", "iv", "delta", "gamma", "theta", "vega", "earnings_date",
]


@dataclass
class ChainResult:
    df: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    provider: str = ""


# ---------------- expiry targeting ----------------

def third_friday(year: int, month: int) -> dt.date:
    d = dt.date(year, month, 15)
    return d + dt.timedelta(days=(4 - d.weekday()) % 7)


def monthly_expiries(dte_min: int, dte_max: int, today: dt.date | None = None) -> list[dt.date]:
    today = today or dt.date.today()
    out, y, m = [], today.year, today.month
    for _ in range(8):
        f = third_friday(y, m)
        if dte_min <= (f - today).days <= dte_max:
            out.append(f)
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def expiry_after(day: dt.date) -> dt.date:
    """First Friday strictly after the earnings date, so the report is inside the option's life."""
    d = day + dt.timedelta(days=1)
    return d + dt.timedelta(days=(4 - d.weekday()) % 7)


# ---------------- normalization ----------------

def _finish(df: pd.DataFrame, r: float) -> pd.DataFrame:
    """Fill mid/spread/IV/greeks consistently and drop junk rows."""
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    df = df.copy()
    for c in ["strike", "spot", "bid", "ask", "last", "volume", "oi", "iv", "delta", "gamma", "theta", "vega"]:
        if c not in df:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["oi"] = df["oi"].fillna(0)
    df["expiration"] = pd.to_datetime(df["expiration"]).dt.date
    today = dt.date.today()
    df["dte"] = [(e - today).days for e in df["expiration"]]
    df = df[df["dte"] >= 1]
    df["mid"] = np.where((df.bid > 0) & (df.ask > 0), (df.bid + df.ask) / 2, np.nan)
    df["spread_pct"] = np.where(df.mid > 0, (df.ask - df.bid) / df.mid, np.nan)

    # IV sanity: recompute from mid when the provider value is missing or absurd.
    T = df["dte"] / 365.0
    bad = df["iv"].isna() | (df["iv"] < 0.02) | (df["iv"] > 5)
    for i in df.index[bad & df.mid.notna()]:
        df.at[i, "iv"] = bs.implied_vol(df.at[i, "mid"], df.at[i, "spot"], df.at[i, "strike"], T[i], r, df.at[i, "type"])

    # Greeks: compute where missing.
    need = df[["delta", "gamma", "theta", "vega"]].isna().any(axis=1) & df["iv"].notna()
    if need.any():
        sub = df[need]
        Ts = sub.dte / 365.0
        for kind in ("call", "put"):
            m = sub.type == kind
            if not m.any():
                continue
            idx = sub.index[m]
            S, K, t, s = sub.spot[m], sub.strike[m], Ts[m], sub.iv[m]
            calc = {
                "delta": bs.delta(S, K, t, r, s, kind), "gamma": bs.gamma(S, K, t, r, s),
                "theta": bs.theta(S, K, t, r, s, kind), "vega": bs.vega(S, K, t, r, s),
            }
            for col, vals in calc.items():  # keep provider values, fill only the gaps
                df.loc[idx, col] = df.loc[idx, col].fillna(pd.Series(np.asarray(vals), index=idx))
    if "earnings_date" not in df:
        df["earnings_date"] = pd.NaT
    df["earnings_date"] = pd.to_datetime(df["earnings_date"], errors="coerce", utc=True)
    for c in COLUMNS:
        if c not in df:
            df[c] = np.nan
    return df[COLUMNS].sort_values(["ticker", "expiration", "type", "strike"]).reset_index(drop=True)


def normalize_apify(items: list[dict], r: float) -> pd.DataFrame:
    rows = []
    for it in items:
        if it.get("rowType", "contract") != "contract" or it.get("strike") is None:
            continue
        rows.append({
            "ticker": it.get("ticker"),
            "symbol": it.get("contractSymbol"),
            "type": str(it.get("type", "")).lower(),
            "strike": it.get("strike"),
            "expiration": it.get("expiration"),
            "spot": it.get("underlyingPrice"),
            "bid": it.get("bid"),
            "ask": it.get("ask"),
            "last": it.get("lastPrice"),
            "volume": it.get("volume"),
            "oi": it.get("openInterest"),
            "iv": it.get("impliedVolatility"),
            "delta": it.get("delta"),
            "gamma": it.get("gamma"),
            "theta": it.get("theta"),
            "vega": it.get("vega"),
            "earnings_date": it.get("earningsDate"),
        })
    return _finish(pd.DataFrame(rows), r)


# ---------------- Apify provider ----------------

def _apify_run(tickers: list[str], exp_from: dt.date, exp_to: dt.date, window_pct: float, max_usd: float) -> list[dict]:
    if not SETTINGS.apify_token:
        raise RuntimeError("APIFY_TOKEN is missing from your .env file.")
    actor = SETTINGS.apify_actor.replace("/", "~")
    url = f"https://api.apify.com/v2/acts/{actor}/run-sync-get-dataset-items"
    payload = {
        "tickers": tickers[:50],
        "mode": "chain",
        "contractTypes": ["calls", "puts"],
        "expirationFrom": exp_from.isoformat(),
        "expirationTo": exp_to.isoformat(),
        "strikeWindowPercent": window_pct,
        "includeGreeks": True,
        "maxItems": 5000,
        "sortBy": "yahooOrder",
    }
    params = {"token": SETTINGS.apify_token, "maxTotalChargeUsd": round(max(max_usd, 0.01), 2), "timeout": 280}
    resp = requests.post(url, params=params, json=payload, timeout=300)
    if resp.status_code >= 400:
        # The actor fails (non-2xx) when filters remove every contract. Treat as empty, not fatal.
        if "removed every contract" in resp.text or "no contracts" in resp.text.lower():
            return []
        raise RuntimeError(f"Apify error {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    return data if isinstance(data, list) else []


def fetch_apify(tickers, targets: dict[dt.date, list[str]], window_pct: float, budget_usd: float) -> ChainResult:
    """targets maps an expiry date to the tickers that need it. One Apify run per expiry date."""
    items, warns, spent = [], [], 0.0
    for exp, tks in sorted(targets.items()):
        remaining = budget_usd - spent
        if remaining <= 0.01:
            warns.append(f"Apify budget of ${budget_usd:.2f} used up. Skipped expiry {exp}.")
            continue
        # Window catches holiday-shifted Thursday expiries.
        got = _apify_run(tks, exp - dt.timedelta(days=2), exp + dt.timedelta(days=1), window_pct, remaining)
        spent += len(got) * APIFY_USD_PER_ROW
        if len(got) == APIFY_FREE_ROW_CAP:
            warns.append(
                f"Apify returned exactly {APIFY_FREE_ROW_CAP} rows for {exp}. That is the free plan cap, so the "
                "chain is cut off. Upgrade Apify or switch the provider to 'Yahoo direct'."
            )
        items.extend(got)
    df = normalize_apify(items, SETTINGS.risk_free_rate)
    return ChainResult(df, warns, spent, "Apify")


def estimate_apify_cost(n_tickers: int, n_expiries: int, window_pct: float) -> float:
    # Rough: liquid names list ~1 strike per 1% of spot near the money, both sides.
    rows = n_tickers * n_expiries * 2 * max(10, 2 * window_pct)
    return rows * APIFY_USD_PER_ROW


# ---------------- Yahoo direct provider (yfinance, free) ----------------

def yahoo_earnings_date(tk) -> pd.Timestamp | None:
    try:
        cal = tk.calendar
        ed = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if ed:
            d = ed[0] if isinstance(ed, (list, tuple)) else ed
            return pd.Timestamp(d, tz="UTC")
    except Exception:
        pass
    return None


def fetch_yahoo(tickers, targets: dict[dt.date, list[str]], window_pct: float) -> ChainResult:
    import yfinance as yf

    by_ticker: dict[str, list[dt.date]] = defaultdict(list)
    for exp, tks in targets.items():
        for t in tks:
            by_ticker[t].append(exp)
    frames, warns = [], []
    for t, exps in by_ticker.items():
        try:
            tk = yf.Ticker(t)
            listed = [dt.date.fromisoformat(x) for x in tk.options]
            if not listed:
                warns.append(f"{t}: no listed options.")
                continue
            spot = float(tk.fast_info["last_price"])
            ed = yahoo_earnings_date(tk)
            chosen = set()
            for e in exps:
                best = min(listed, key=lambda x: abs((x - e).days))
                if abs((best - e).days) <= 3:
                    chosen.add(best)
            for e in sorted(chosen):
                oc = tk.option_chain(e.isoformat())
                for kind, part in (("call", oc.calls), ("put", oc.puts)):
                    p = part.copy()
                    p = p[(p.strike >= spot * (1 - window_pct / 100)) & (p.strike <= spot * (1 + window_pct / 100))]
                    frames.append(pd.DataFrame({
                        "ticker": t, "symbol": p.contractSymbol, "type": kind, "strike": p.strike,
                        "expiration": e, "spot": spot, "bid": p.bid, "ask": p.ask, "last": p.lastPrice,
                        "volume": p.volume, "oi": p.openInterest, "iv": p.impliedVolatility,
                        "earnings_date": ed,
                    }))
            time.sleep(0.3)
        except Exception as exc:  # one bad ticker should not kill the scan
            warns.append(f"{t}: {exc}")
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    # Yahoo's own IV on illiquid strikes is unreliable. Force a recompute from mid.
    if not df.empty:
        df["iv"] = np.nan
    return ChainResult(_finish(df, SETTINGS.risk_free_rate), warns, 0.0, "Yahoo direct")


def upcoming_earnings_yahoo(tickers) -> dict[str, pd.Timestamp]:
    import yfinance as yf

    out = {}
    for t in tickers:
        ed = yahoo_earnings_date(yf.Ticker(t))
        if ed is not None:
            out[t] = ed
    return out


# ---------------- orchestration ----------------

def scan_chains(tickers: list[str], dte_min: int, dte_max: int, earnings_days: int, window_pct: float,
                provider: str, budget_usd: float) -> ChainResult:
    """Phase 1: monthly expiries in the DTE window. Phase 2: the expiry right after each upcoming earnings."""
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    months = monthly_expiries(dte_min, dte_max)
    if not months:
        months = monthly_expiries(dte_min, dte_max + 21)[:1]
    targets = {m: list(tickers) for m in months}
    today = dt.date.today()

    if provider == "yahoo":
        earn = upcoming_earnings_yahoo(tickers)
        for t, ts in earn.items():
            d = ts.date()
            if 0 <= (d - today).days <= earnings_days:
                targets.setdefault(expiry_after(d), []).append(t)
        return fetch_yahoo(tickers, targets, max(window_pct, 15) if len(targets) > len(months) else window_pct)

    res = fetch_apify(tickers, targets, window_pct, budget_usd)
    # Earnings dates arrive with the phase 1 rows.
    phase2: dict[dt.date, list[str]] = defaultdict(list)
    if not res.df.empty:
        ed = res.df.groupby("ticker")["earnings_date"].first().dropna()
        for t, ts in ed.items():
            d = ts.date()
            exp = expiry_after(d)
            if 0 <= (d - today).days <= earnings_days and exp not in months:
                phase2[exp].append(t)
    if phase2:
        more = fetch_apify(tickers, dict(phase2), max(window_pct, 15), budget_usd - res.cost_usd)
        res = ChainResult(pd.concat([res.df, more.df], ignore_index=True).drop_duplicates("symbol"),
                          res.warnings + more.warnings, res.cost_usd + more.cost_usd, "Apify")
    return res


# ---------------- Alpaca provider (free paper account, works from cloud servers) ----------------

ALPACA_DATA = "https://data.alpaca.markets"
ALPACA_TRADE = "https://paper-api.alpaca.markets"


def _alpaca_get(url: str, key: str, secret: str, params: dict) -> dict:
    h = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret, "accept": "application/json"}
    for attempt in range(3):
        r = requests.get(url, headers=h, params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(2 + attempt * 3)
            continue
        if r.status_code in (401, 403):
            raise RuntimeError("Alpaca rejected the API keys. Check the key and secret.")
        r.raise_for_status()
        return r.json()
    raise RuntimeError("Alpaca rate limit hit. Wait a minute and scan again.")


def _alpaca_paged(url, key, secret, params, list_key):
    out, token = ([] if list_key != "snapshots" else {}), None
    for _ in range(20):
        p = dict(params)
        if token:
            p["page_token"] = token
        d = _alpaca_get(url, key, secret, p)
        part = d.get(list_key) or ([] if list_key != "snapshots" else {})
        if isinstance(out, dict):
            out.update(part)
        else:
            out.extend(part)
        token = d.get("next_page_token")
        if not token:
            break
    return out


def alpaca_spots(tickers, key, secret) -> dict:
    d = _alpaca_get(f"{ALPACA_DATA}/v2/stocks/snapshots", key, secret,
                    {"symbols": ",".join(tickers), "feed": "iex"})
    out = {}
    for t, snap in (d.items() if isinstance(d, dict) else []):
        px = ((snap or {}).get("latestTrade") or {}).get("p") or ((snap or {}).get("dailyBar") or {}).get("c")
        if px:
            out[t] = float(px)
    return out


def fetch_alpaca(tickers, targets: dict, window_pct: float, key: str, secret: str) -> ChainResult:
    by_ticker = defaultdict(list)
    for exp, tks in targets.items():
        for t in tks:
            by_ticker[t].append(exp)
    warns, frames = [], []
    spots = alpaca_spots(list(by_ticker), key, secret)
    for t, exps in by_ticker.items():
        S = spots.get(t)
        if not S:
            warns.append(f"{t}: no stock price from Alpaca.")
            continue
        lo, hi = round(S * (1 - window_pct / 100), 2), round(S * (1 + window_pct / 100), 2)
        try:
            contracts = _alpaca_paged(f"{ALPACA_TRADE}/v2/options/contracts", key, secret, {
                "underlying_symbols": t, "status": "active", "limit": 1000,
                "expiration_date_gte": (min(exps) - dt.timedelta(days=3)).isoformat(),
                "expiration_date_lte": (max(exps) + dt.timedelta(days=3)).isoformat(),
                "strike_price_gte": lo, "strike_price_lte": hi}, "option_contracts")
        except Exception as exc:
            warns.append(f"{t}: {exc}")
            continue
        if not contracts:
            warns.append(f"{t}: no listed options near the target dates.")
            continue
        listed = sorted({dt.date.fromisoformat(c["expiration_date"]) for c in contracts})
        chosen = set()
        for e in exps:
            best = min(listed, key=lambda x: abs((x - e).days))
            if abs((best - e).days) <= 3:
                chosen.add(best)
        oi = {c["symbol"]: c.get("open_interest") for c in contracts}
        for e in sorted(chosen):
            try:
                snaps = _alpaca_paged(f"{ALPACA_DATA}/v1beta1/options/snapshots/{t}", key, secret, {
                    "feed": "indicative", "expiration_date": e.isoformat(), "limit": 1000,
                    "strike_price_gte": lo, "strike_price_lte": hi}, "snapshots")
            except Exception as exc:
                warns.append(f"{t} {e}: {exc}")
                continue
            rows = []
            for c in contracts:
                if dt.date.fromisoformat(c["expiration_date"]) != e:
                    continue
                sn = snaps.get(c["symbol"]) or {}
                q, g = sn.get("latestQuote") or {}, sn.get("greeks") or {}
                bar = sn.get("dailyBar") or {}
                rows.append({
                    "ticker": t, "symbol": c["symbol"], "type": c.get("type"), "strike": float(c["strike_price"]),
                    "expiration": e, "spot": S, "bid": q.get("bp"), "ask": q.get("ap"),
                    "last": (sn.get("latestTrade") or {}).get("p"), "volume": bar.get("v"),
                    "oi": oi.get(c["symbol"]), "iv": sn.get("impliedVolatility"), "delta": g.get("delta"),
                })
            if rows:
                frames.append(pd.DataFrame(rows))
            time.sleep(0.2)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not df.empty:
        earn = {}
        try:
            earn = upcoming_earnings_yahoo(list(df.ticker.unique()))
        except Exception:
            pass
        df["earnings_date"] = df.ticker.map(earn)
        if not earn:
            warns.append("Earnings dates unavailable from this server, so the Earnings tab may be empty.")
    return ChainResult(_finish(df, SETTINGS.risk_free_rate), warns, 0.0, "Alpaca")


def scan_chains_alpaca(tickers, dte_min, dte_max, earnings_days, window_pct, key, secret) -> ChainResult:
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    months = monthly_expiries(dte_min, dte_max) or monthly_expiries(dte_min, dte_max + 21)[:1]
    targets = {m: list(tickers) for m in months}
    today = dt.date.today()
    try:
        for t, ts in upcoming_earnings_yahoo(tickers).items():
            d = ts.date()
            if 0 <= (d - today).days <= earnings_days:
                targets.setdefault(expiry_after(d), []).append(t)
    except Exception:
        pass
    wide = max(window_pct, 15) if len(targets) > len(months) else window_pct
    return fetch_alpaca(tickers, targets, wide, key, secret)
