"""Plain-English sentences for the top results, written to be read aloud by text to speech."""
from __future__ import annotations

import math

import pandas as pd


def _money(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"${x:,.0f}"


def _date(d) -> str:
    return pd.Timestamp(d).strftime("%B %-d")


def premium(row: pd.Series) -> str:
    edge = "positive" if row.ev > 0 else "negative"
    sz = (f"Your risk limit allows {int(row.contracts)} contract{'s' if row.contracts != 1 else ''}."
          if row.contracts else "This trade is too large for your risk limit.")
    flag = " Warning: earnings fall before expiration." if row.get("flag") else ""
    return (f"{row.ticker}. {row.strategy}, expiring {_date(row.expiration)}, {int(row.dte)} days out. "
            f"{row.legs}. You collect about {_money(row.credit)} per contract and can lose up to "
            f"{_money(row.max_loss)}. Chance of profit about {row.prob_profit:.0%}. "
            f"Implied volatility is {row.iv_hv:.2f} times recent realized volatility. "
            f"Expected value is {edge}, about {_money(row.ev)} per contract. {sz}{flag}")


def directional(row: pd.Series) -> str:
    sz = (f"Your risk limit allows {int(row.contracts)} contract{'s' if row.contracts != 1 else ''}."
          if row.contracts else "This trade is too large for your risk limit.")
    return (f"{row.ticker} trend is {row.trend}. {row.strategy}, expiring {_date(row.expiration)}. {row.legs}. "
            f"Costs about {_money(row.debit)} per contract, which is your maximum loss. "
            f"The stock must move {row.be_move_pct:.1%} to break even, which is "
            f"{row.be_vs_expected:.2f} times the move the market expects. Chance of profit about {row.prob_profit:.0%}. "
            f"Expected value about {_money(row.ev)} per contract, assuming no directional edge. {sz}")


def earnings(row: pd.Series) -> str:
    base = f"{row.ticker} reports on {_date(row.earnings_date)}. Verdict: {row.verdict}. {row.why}"
    if row.legs:
        base += f" Trade: {row.legs}, expiring {_date(row.expiration)}. Max loss {_money(row.max_loss)} per contract."
    return base
