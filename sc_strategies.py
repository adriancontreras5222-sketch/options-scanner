"""Trade scanners: premium selling, directional buys, and earnings plays.

Every candidate is scored the same honest way:
  * Fill prices assume you give up part of the bid/ask spread (slippage setting).
  * EV and probability of profit use a lognormal with ZERO drift and YOUR volatility forecast
    (realized vol), not the option's implied vol. Pricing with implied vol would make every
    trade look fair by construction. The edge is the gap between implied and realized.
  * Size = how many contracts fit your risk-per-trade limit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

import sc_bs as bs


@dataclass
class Rules:
    account: float = 25_000
    risk_pct: float = 0.02          # max loss per trade as share of account
    max_csp_alloc: float = 0.25     # max capital in one cash-secured put
    min_oi: int = 500
    min_volume: int = 10
    max_spread_pct: float = 0.12    # bid/ask width as share of mid
    slippage: float = 0.5           # 0 = fill at mid, 1 = fill at bid/ask
    short_delta_lo: float = 0.15
    short_delta_hi: float = 0.35
    max_width_steps: int = 3
    min_credit_to_width: float = 0.20
    vol_forecast: str = "blend"     # 'hv20' | 'hv60' | 'blend'


# ---------------- helpers ----------------

def liquid(df: pd.DataFrame, rules: Rules) -> pd.DataFrame:
    return df[(df.bid > 0) & (df.ask > df.bid) & (df.oi >= rules.min_oi) & (df.volume >= rules.min_volume)
              & (df.spread_pct <= rules.max_spread_pct)]


def sell_px(row, s):
    return row.mid - s * (row.mid - row.bid)


def buy_px(row, s):
    return row.mid + s * (row.ask - row.mid)


def forecast_vol(profile: dict, rules: Rules) -> float:
    key = {"hv20": "hv20", "hv60": "hv60"}.get(rules.vol_forecast, "hv_forecast")
    return profile.get(key, float("nan"))


def size(max_loss_per_contract: float, rules: Rules) -> int:
    if max_loss_per_contract <= 0:
        return 0
    return int((rules.account * rules.risk_pct) // max_loss_per_contract)


def crosses_earnings(exp, earn_ts) -> bool:
    if earn_ts is None or pd.isna(earn_ts):
        return False
    d = earn_ts.date()
    return pd.Timestamp.today().date() <= d <= exp


def _common(row: dict, T: float, iv: float, hv: float):
    row["iv_hv"] = iv / hv if hv and hv > 0 and iv == iv else float("nan")
    row["years"] = T
    return row


# ---------------- premium selling ----------------

def premium_candidates(chain: pd.DataFrame, profiles: dict, rules: Rules, include_earnings: bool = False,
                       owned: set[str] | None = None) -> pd.DataFrame:
    owned = owned or set()
    out = []
    s = rules.slippage
    for (tkr, exp), g in chain.groupby(["ticker", "expiration"]):
        prof = profiles.get(tkr) or {}
        hv = forecast_vol(prof, rules)
        if not hv or math.isnan(hv):
            continue
        earn = g.earnings_date.iloc[0]
        if crosses_earnings(exp, earn) and not include_earnings:
            continue
        S, dte = float(g.spot.iloc[0]), int(g.dte.iloc[0])
        T = dte / 365
        L = liquid(g, rules)
        puts = L[L.type == "put"].sort_values("strike")
        calls = L[L.type == "call"].sort_values("strike")
        flag = "EARNINGS INSIDE" if crosses_earnings(exp, earn) else ""

        def band(df):
            return df[(df.delta.abs() >= rules.short_delta_lo) & (df.delta.abs() <= rules.short_delta_hi)]

        # Cash-secured puts
        for _, p in band(puts).iterrows():
            cr = sell_px(p, s)
            K = p.strike
            ev, pop = bs.payoff_stats(lambda x: cr - np.maximum(K - x, 0), S, T, hv)
            max_loss = (K - cr) * 100
            capital = K * 100
            n = 1 if capital <= rules.account * rules.max_csp_alloc else 0
            out.append(_common({
                "ticker": tkr, "strategy": "Cash-secured put", "expiration": exp, "dte": dte,
                "legs": f"Sell {K:g}P", "credit": cr * 100, "max_loss": max_loss, "capital": capital,
                "roc": cr / (K - cr), "ann_roc": cr / (K - cr) * 365 / dte, "prob_profit": pop, "ev": ev * 100,
                "ev_per_risk": ev * 100 / capital, "short_delta": p.delta, "iv": p.iv,
                "breakeven": K - cr, "contracts": n, "flag": flag, "spot": S,
            }, T, p.iv, hv))

        # Covered calls (only for tickers you say you own)
        if tkr in owned:
            for _, c in band(calls).iterrows():
                cr = sell_px(c, s)
                K = c.strike
                ev, pop = bs.payoff_stats(lambda x: cr - np.maximum(x - K, 0), S, T, hv)
                out.append(_common({
                    "ticker": tkr, "strategy": "Covered call", "expiration": exp, "dte": dte,
                    "legs": f"Own 100 sh, sell {K:g}C", "credit": cr * 100, "max_loss": float("nan"),
                    "capital": S * 100, "roc": cr / S, "ann_roc": cr / S * 365 / dte, "prob_profit": pop,
                    "ev": ev * 100, "ev_per_risk": ev / S, "short_delta": c.delta, "iv": c.iv,
                    "breakeven": S - cr, "contracts": 1, "flag": flag, "spot": S,
                }, T, c.iv, hv))

        # Vertical credit spreads
        verticals = []
        for kind, shorts, wings in (("put", band(puts), puts), ("call", band(calls), calls)):
            strikes = list(wings.strike)
            for _, sh in shorts.iterrows():
                i = strikes.index(sh.strike)
                for step in range(1, rules.max_width_steps + 1):
                    j = i - step if kind == "put" else i + step
                    if j < 0 or j >= len(strikes):
                        break
                    lg = wings.iloc[j]
                    cr = sell_px(sh, s) - buy_px(lg, s)
                    w = abs(sh.strike - lg.strike)
                    if cr <= 0 or cr / w < rules.min_credit_to_width:
                        continue
                    Ks, Kl = sh.strike, lg.strike
                    if kind == "put":
                        pay = lambda x, cr=cr, Ks=Ks, Kl=Kl: cr - np.clip(Ks - x, 0, Ks - Kl)  # noqa: E731
                        be = Ks - cr
                    else:
                        pay = lambda x, cr=cr, Ks=Ks, Kl=Kl: cr - np.clip(x - Ks, 0, Kl - Ks)  # noqa: E731
                        be = Ks + cr
                    ev, pop = bs.payoff_stats(pay, S, T, hv)
                    ml = (w - cr) * 100
                    rec = _common({
                        "ticker": tkr, "strategy": f"{kind.title()} credit spread", "expiration": exp, "dte": dte,
                        "legs": f"Sell {Ks:g}{kind[0].upper()} / Buy {Kl:g}{kind[0].upper()}",
                        "credit": cr * 100, "max_loss": ml, "capital": ml, "roc": cr / (w - cr),
                        "ann_roc": cr / (w - cr) * 365 / dte, "prob_profit": pop, "ev": ev * 100,
                        "ev_per_risk": ev * 100 / ml, "short_delta": sh.delta, "iv": sh.iv, "breakeven": be,
                        "contracts": size(ml, rules), "flag": flag, "spot": S,
                    }, T, sh.iv, hv)
                    rec["_k"] = (kind, Ks, Kl, cr)
                    verticals.append(rec)
        out.extend(verticals)

        # Iron condors: pair the best put spread with the best call spread of similar short delta
        ps = [v for v in verticals if v["_k"][0] == "put"]
        cs = [v for v in verticals if v["_k"][0] == "call"]
        ps = sorted(ps, key=lambda v: -v["ev_per_risk"])[:6]
        cs = sorted(cs, key=lambda v: -v["ev_per_risk"])[:6]
        for p_ in ps:
            for c_ in cs:
                _, pKs, pKl, pcr = p_["_k"]
                _, cKs, cKl, ccr = c_["_k"]
                if abs(abs(p_["short_delta"]) - abs(c_["short_delta"])) > 0.10:
                    continue
                cr = pcr + ccr
                w = max(pKs - pKl, cKl - cKs)
                pay = lambda x, a=(pKs, pKl, cKs, cKl, cr): a[4] - np.clip(a[0] - x, 0, a[0] - a[1]) - np.clip(x - a[2], 0, a[3] - a[2])  # noqa: E731
                ev, pop = bs.payoff_stats(pay, S, T, hv)
                ml = (w - cr) * 100
                if ml <= 0:
                    continue
                out.append(_common({
                    "ticker": tkr, "strategy": "Iron condor", "expiration": exp, "dte": dte,
                    "legs": f"{pKl:g}/{pKs:g}P  {cKs:g}/{cKl:g}C", "credit": cr * 100, "max_loss": ml,
                    "capital": ml, "roc": cr / (w - cr), "ann_roc": cr / (w - cr) * 365 / dte, "prob_profit": pop,
                    "ev": ev * 100, "ev_per_risk": ev * 100 / ml,
                    "short_delta": (abs(p_["short_delta"]) + abs(c_["short_delta"])) / 2,
                    "iv": (p_["iv"] + c_["iv"]) / 2, "breakeven": f"{pKs - cr:.2f} / {cKs + cr:.2f}",
                    "contracts": size(ml, rules), "flag": flag, "spot": S,
                }, T, (p_["iv"] + c_["iv"]) / 2, hv))
    df = pd.DataFrame(out)
    if df.empty:
        return df
    df = df.drop(columns=[c for c in ["_k"] if c in df])
    return df.sort_values("ev_per_risk", ascending=False).reset_index(drop=True)


# ---------------- directional buys ----------------

def directional_candidates(chain: pd.DataFrame, profiles: dict, rules: Rules,
                           include_earnings: bool = False) -> pd.DataFrame:
    out = []
    s = rules.slippage
    for (tkr, exp), g in chain.groupby(["ticker", "expiration"]):
        prof = profiles.get(tkr) or {}
        trend = prof.get("trend", "neutral")
        hv = forecast_vol(prof, rules)
        if trend == "neutral" or not hv or math.isnan(hv):
            continue
        earn = g.earnings_date.iloc[0]
        if crosses_earnings(exp, earn) and not include_earnings:
            continue
        S, dte = float(g.spot.iloc[0]), int(g.dte.iloc[0])
        T = dte / 365
        kind = "call" if trend == "bullish" else "put"
        L = liquid(g, rules)
        side = L[L.type == kind].sort_values("strike")
        strikes = list(side.strike)
        exp_move = S * float(side.iv.median()) * math.sqrt(T) if not side.empty else float("nan")
        flag = "EARNINGS INSIDE" if crosses_earnings(exp, earn) else ""
        for _, o in side[(side.delta.abs() >= 0.40) & (side.delta.abs() <= 0.70)].iterrows():
            debit = buy_px(o, s)
            K = o.strike
            if kind == "call":
                pay = lambda x, K=K, d=debit: np.maximum(x - K, 0) - d  # noqa: E731
                be = K + debit
            else:
                pay = lambda x, K=K, d=debit: np.maximum(K - x, 0) - d  # noqa: E731
                be = K - debit
            ev, pop = bs.payoff_stats(pay, S, T, hv)
            out.append(_common({
                "ticker": tkr, "trend": trend, "strategy": f"Long {kind}", "expiration": exp, "dte": dte,
                "legs": f"Buy {K:g}{kind[0].upper()}", "debit": debit * 100, "max_loss": debit * 100,
                "max_gain": float("inf"), "reward_risk": float("nan"), "breakeven": be,
                "be_move_pct": abs(be / S - 1), "be_vs_expected": abs(be - S) / exp_move if exp_move else float("nan"),
                "prob_profit": pop, "ev": ev * 100, "ev_per_risk": ev / debit, "delta": o.delta, "iv": o.iv,
                "contracts": size(debit * 100, rules), "flag": flag, "spot": S,
            }, T, o.iv, hv))
            # Debit spreads from the same long leg
            i = strikes.index(K)
            for step in range(1, rules.max_width_steps + 2):
                j = i + step if kind == "call" else i - step
                if j < 0 or j >= len(strikes):
                    break
                sh = side.iloc[j]
                d2 = debit - sell_px(sh, s)
                w = abs(sh.strike - K)
                if d2 <= 0 or d2 >= w:
                    continue
                Ks = sh.strike
                if kind == "call":
                    pay2 = lambda x, K=K, Ks=Ks, d=d2: np.clip(x - K, 0, Ks - K) - d  # noqa: E731
                    be2 = K + d2
                else:
                    pay2 = lambda x, K=K, Ks=Ks, d=d2: np.clip(K - x, 0, K - Ks) - d  # noqa: E731
                    be2 = K - d2
                ev2, pop2 = bs.payoff_stats(pay2, S, T, hv)
                out.append(_common({
                    "ticker": tkr, "trend": trend, "strategy": f"{kind.title()} debit spread", "expiration": exp,
                    "dte": dte, "legs": f"Buy {K:g} / Sell {Ks:g}{kind[0].upper()}", "debit": d2 * 100,
                    "max_loss": d2 * 100, "max_gain": (w - d2) * 100, "reward_risk": (w - d2) / d2,
                    "breakeven": be2, "be_move_pct": abs(be2 / S - 1),
                    "be_vs_expected": abs(be2 - S) / exp_move if exp_move else float("nan"),
                    "prob_profit": pop2, "ev": ev2 * 100, "ev_per_risk": ev2 / d2, "delta": o.delta - sh.delta,
                    "iv": o.iv, "contracts": size(d2 * 100, rules), "flag": flag, "spot": S,
                }, T, o.iv, hv))
    df = pd.DataFrame(out)
    if df.empty:
        return df
    return df.sort_values("ev_per_risk", ascending=False).reset_index(drop=True)


# ---------------- earnings ----------------

def earnings_candidates(chain: pd.DataFrame, history: dict, rules: Rules,
                        sell_ratio: float = 1.25, buy_ratio: float = 0.85) -> pd.DataFrame:
    """history[ticker] = list of {'date','timing','move'} for past reports."""
    out = []
    s = rules.slippage
    today = pd.Timestamp.today().date()
    for tkr, g in chain.groupby("ticker"):
        earn = g.earnings_date.dropna()
        if earn.empty:
            continue
        ed = earn.iloc[0].date()
        if ed < today:
            continue
        exps = sorted(e for e in g.expiration.unique() if e >= ed)
        if not exps:
            continue
        exp = exps[0]
        x = g[(g.expiration == exp) & (g.bid > 0) & (g.ask > 0)]
        if x.empty:
            continue
        S = float(x.spot.iloc[0])
        k = x.iloc[(x.strike - S).abs().argsort().iloc[0]].strike
        c = x[(x.strike == k) & (x.type == "call")]
        p = x[(x.strike == k) & (x.type == "put")]
        if c.empty or p.empty:
            continue
        c, p = c.iloc[0], p.iloc[0]
        straddle = c.mid + p.mid
        implied = straddle / S
        moves = [abs(m["move"]) for m in history.get(tkr, [])]
        n = len(moves)
        hist_avg = float(np.mean(moves)) if n else float("nan")
        exceed = float(np.mean([m > implied for m in moves])) if n else float("nan")
        ratio = implied / hist_avg if n and hist_avg > 0 else float("nan")
        liquid_ok = (min(c.oi, p.oi) >= rules.min_oi) and (max(c.spread_pct, p.spread_pct) <= rules.max_spread_pct)

        if n < 4:
            verdict, why = "No trade", f"Only {n} past reports found. Not enough history to trust."
        elif not liquid_ok:
            verdict, why = "No trade", "ATM options fail your liquidity rules."
        elif ratio >= sell_ratio and exceed <= 0.25:
            verdict, why = "Sell premium", (f"Market prices a {implied:.1%} move. Stock averaged {hist_avg:.1%}. "
                                            f"Beat the implied move {exceed:.0%} of the time.")
        elif ratio <= buy_ratio or exceed >= 0.5:
            verdict, why = "Buy straddle", (f"Market prices only {implied:.1%}. Stock averaged {hist_avg:.1%} "
                                            f"and beat the implied move {exceed:.0%} of the time.")
        else:
            verdict, why = "No trade", f"Implied {implied:.1%} vs history {hist_avg:.1%}. No clear mispricing."

        legs, debit_credit, max_loss, contracts = "", float("nan"), float("nan"), 0
        if verdict == "Buy straddle":
            d = buy_px(c, s) + buy_px(p, s)
            legs, debit_credit, max_loss = f"Buy {k:g}C + {k:g}P", -d * 100, d * 100
            contracts = size(max_loss, rules)
        elif verdict == "Sell premium":
            # Defined risk: short strikes just outside the implied move, wings one more strike out.
            puts = x[x.type == "put"].sort_values("strike")
            calls = x[x.type == "call"].sort_values("strike")
            ps = puts[puts.strike <= S - straddle]
            cs = calls[calls.strike >= S + straddle]
            if len(ps) >= 2 and len(cs) >= 2:
                sp, lp = ps.iloc[-1], ps.iloc[-2]
                sc, lc = cs.iloc[0], cs.iloc[1]
                cr = sell_px(sp, s) - buy_px(lp, s) + sell_px(sc, s) - buy_px(lc, s)
                w = max(sp.strike - lp.strike, lc.strike - sc.strike)
                if cr > 0 and w > cr:
                    legs = f"Iron condor {lp.strike:g}/{sp.strike:g}P  {sc.strike:g}/{lc.strike:g}C"
                    debit_credit, max_loss = cr * 100, (w - cr) * 100
                    contracts = size(max_loss, rules)
            if not legs:
                verdict, why = "No trade", why + " But no liquid strikes outside the implied move."
        out.append({
            "ticker": tkr, "earnings_date": ed, "expiration": exp, "spot": S, "atm_strike": k,
            "straddle": straddle * 100, "implied_move": implied, "hist_avg_move": hist_avg, "ratio": ratio,
            "beat_rate": exceed, "reports": n, "verdict": verdict, "why": why, "legs": legs,
            "credit(+)/debit(-)": debit_credit, "max_loss": max_loss, "contracts": contracts,
        })
    df = pd.DataFrame(out)
    if df.empty:
        return df
    order = {"Sell premium": 0, "Buy straddle": 1, "No trade": 2}
    return df.sort_values(["verdict", "ratio"], key=lambda col: col.map(order) if col.name == "verdict" else col,
                          ascending=[True, False]).reset_index(drop=True)
