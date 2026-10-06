"""Options Scanner, public web version. Free Yahoo data, no keys, one private session per visitor."""
from __future__ import annotations

import datetime as dt
import json
import math

import pandas as pd
import streamlit as st

import sc_chain as data_chain, sc_history as data_history, sc_journal as journal, sc_narrate as narrate
import sc_strategies as strategies, sc_vol as vol
from sc_config import SETTINGS

MAX_TICKERS = 15

st.set_page_config(page_title="Options Scanner", layout="wide")

# ---------------- accessibility: big text, high contrast ----------------
size = st.sidebar.slider("Text size", 16, 32, 22, help="Makes every label, number and table larger.")
st.markdown(f"""
<style>
html, body, [class*="css"], .stMarkdown, .stText, label, p, li, td, th, input, textarea, select, button
  {{ font-size: {size}px !important; line-height: 1.45; }}
h1 {{ font-size: {size * 1.8}px !important; }}
h2 {{ font-size: {size * 1.45}px !important; }}
h3 {{ font-size: {size * 1.2}px !important; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ padding: 8px 12px !important; border-bottom: 1px solid #555 !important; text-align: left !important; }}
th {{ background: #1f2a44 !important; color: #ffffff !important; }}
.pick {{ border-left: 8px solid #ffd400; padding: 12px 18px; margin: 10px 0; background: #111a2e; }}
.warn {{ border-left: 8px solid #ff5c5c; padding: 12px 18px; margin: 10px 0; background: #2a1111; }}
button:focus, input:focus {{ outline: 4px solid #ffd400 !important; }}
.stApp, [data-testid="stHeader"] {{ background: #0b1020 !important; color: #ffffff !important; }}
[data-testid="stSidebar"] {{ background: #1a2340 !important; }}
.stApp *, [data-testid="stSidebar"] * {{ color: #ffffff; }}
input, textarea, [data-baseweb="select"] > div, [data-baseweb="input"] > div {{ background: #0b1020 !important; color: #ffffff !important; }}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {{ background: #ffd400 !important; color: #000000 !important; border: none !important; }}
.stButton > button[kind="primary"] *, .stDownloadButton > button[kind="primary"] * {{ color: #000000 !important; }}
</style>""", unsafe_allow_html=True)


def speak_button(text: str, key: str):
    """Reads text aloud with the browser's built-in speech. Nothing is sent anywhere."""
    payload = json.dumps(text)
    html = f"""
    <button id="b{key}" style="font-size:20px;padding:10px 18px;background:#ffd400;border:none;
      border-radius:6px;cursor:pointer;font-weight:700">Read aloud</button>
    <button id="s{key}" style="font-size:20px;padding:10px 18px;margin-left:8px;border-radius:6px;
      cursor:pointer">Stop</button>
    <script>
      document.getElementById("b{key}").onclick = () => {{
        speechSynthesis.cancel(); const u = new SpeechSynthesisUtterance({payload}); u.rate = 1.0;
        speechSynthesis.speak(u); }};
      document.getElementById("s{key}").onclick = () => speechSynthesis.cancel();
    </script>"""
    if hasattr(st, "iframe"):
        st.iframe(html, height=60)
    else:  # older Streamlit
        import streamlit.components.v1 as components
        components.html(html, height=60)


def box(text: str, cls: str = "pick", lead: str = ""):
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace("$", "&#36;")
    st.markdown(f'<div class="{cls}">{lead}{safe}</div>', unsafe_allow_html=True)


def pct(x, d=0):
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}%}"


def money(x):
    if x is None or (isinstance(x, float) and (math.isnan(x))):
        return ""
    if isinstance(x, float) and math.isinf(x):
        return "Unlimited"
    return f"${x:,.0f}"


HEAD = {"dte": "Days", "prob_profit": "Win chance", "ev": "Exp. value", "ev_per_risk": "EV / risk",
        "iv_hv": "IV / HV", "be_move_pct": "Move to break even", "be_vs_expected": "BE / expected move",
        "hist_avg_move": "Past avg move", "beat_rate": "Past beat rate", "credit(+)/debit(-)": "Credit(+) Debit(-)",
        "atm_iv": "ATM IV", "hv20": "HV 20d", "hv60": "HV 60d", "iv_vs_hv": "IV / HV", "iv_rank": "IV rank",
        "ret20": "20d return", "sma50": "50d avg", "sma200": "200d avg", "rsi": "RSI"}


def big_table(df: pd.DataFrame, formats: dict, n: int):
    """HTML table so the text-size slider applies (the interactive grid ignores CSS)."""
    t = df.head(n).copy()
    for col, f in formats.items():
        if col in t:
            t[col] = t[col].map(f)
    t.index = range(1, len(t) + 1)
    t.columns = [HEAD.get(c, c.replace("_", " ").capitalize()) for c in t.columns]
    st.markdown(f'<div style="overflow-x:auto">{t.to_html(escape=True).replace("$", "&#36;")}</div>',
                unsafe_allow_html=True)


# ---------------- sidebar ----------------
st.sidebar.header("Scan settings")
tick_text = st.sidebar.text_area("Tickers (space separated)", SETTINGS.watchlist, height=110)
tickers = list(dict.fromkeys(t.upper() for t in tick_text.replace(",", " ").split() if t.strip()))
if len(tickers) > MAX_TICKERS:
    st.sidebar.warning(f"Using the first {MAX_TICKERS} tickers. Scans are capped to keep the free data source working.")
    tickers = tickers[:MAX_TICKERS]
c1, c2 = st.sidebar.columns(2)
dte_min = c1.number_input("Min days", 7, 120, 21)
dte_max = c2.number_input("Max days", 14, 180, 50)
earn_days = st.sidebar.number_input("Earnings within (days)", 0, 60, 21)
window = st.sidebar.slider("Strike range around price (%)", 5, 30, 12)

st.sidebar.header("Your risk rules")
account = st.sidebar.number_input("Account size ($)", 1_000, 10_000_000, 25_000, 1_000)
risk_pct = st.sidebar.slider("Max loss per trade (% of account)", 0.5, 10.0, 2.0, 0.5) / 100
min_oi = st.sidebar.number_input("Min open interest", 0, 100_000, 500, 50)
min_vol = st.sidebar.number_input("Min volume today", 0, 100_000, 10, 5)
max_spr = st.sidebar.slider("Max bid/ask spread (% of price)", 2, 40, 12) / 100
slip = st.sidebar.slider("Slippage (0 = mid, 1 = worst side)", 0.0, 1.0, 0.5, 0.05)
d_lo, d_hi = st.sidebar.slider("Short strike delta range", 0.05, 0.50, (0.15, 0.35), 0.01)
vf = st.sidebar.selectbox("Volatility forecast", ["blend", "hv20", "hv60"],
                          help="Realized volatility used to judge whether options are cheap or rich.")
owned = st.sidebar.text_input("Tickers you own 100+ shares of (for covered calls)", "")
rules = strategies.Rules(account=account, risk_pct=risk_pct, min_oi=min_oi, min_volume=min_vol,
                         max_spread_pct=max_spr, slippage=slip, short_delta_lo=d_lo, short_delta_hi=d_hi,
                         vol_forecast=vf)

# ---------------- header & scan ----------------
st.title("Options Scanner")
box("Educational tool, not investment advice. Data is free, delayed Yahoo Finance data and can be wrong or missing. "
    "Options can lose more than you expect. Check every price with your broker before trading.", "warn")
months = data_chain.monthly_expiries(dte_min, dte_max)
st.write(f"Scanning {len(tickers)} tickers. Monthly expiries: {', '.join(m.strftime('%b %d') for m in months) or 'none in range'}, "
         "plus the expiry after any upcoming earnings. A scan takes about a minute.")


@st.cache_data(ttl=900, show_spinner=False, max_entries=50)
def run_scan(tks: tuple, dmin: int, dmax: int, edays: int, win: int) -> dict:
    """Shared across visitors for 15 minutes so identical scans do not hit Yahoo twice."""
    res = data_chain.scan_chains(list(tks), dmin, dmax, edays, win, "yahoo", 0.0)
    chain = res.df
    profiles, history, warns = {}, {}, list(res.warnings)
    for t in tks:
        bars, _ = data_history.get_bars(t, use_massive=False)
        if bars is None or bars.empty:
            warns.append(f"{t}: no price history from Yahoo.")
        profiles[t] = vol.stock_profile(bars)
        ct = chain[chain.ticker == t]
        if not ct.empty and profiles[t]:
            cur = vol.atm_iv(ct)
            vol.record_iv(t, cur)
            rank, ndays = vol.iv_rank(t, cur)
            profiles[t].update({"atm_iv": cur, "iv_rank": rank, "iv_days": ndays})
            if ct.earnings_date.notna().any():
                ed = ct.earnings_date.dropna().iloc[0].date()
                if 0 <= (ed - dt.date.today()).days <= edays:
                    history[t] = vol.earnings_moves(bars, data_history.past_earnings(t))
    if chain.empty:
        warns.append("Yahoo returned no option data. It may be rate limiting this server. Wait a few minutes and try again.")
    return {"time": dt.datetime.now(), "chain": chain, "profiles": profiles, "history": history,
            "warnings": warns, "cost": 0.0, "provider": "Yahoo"}


if st.button("Run scan", type="primary", width="stretch", disabled=not tickers):
    with st.spinner("Fetching option chains and price history. About a minute..."):
        try:
            st.session_state["scan"] = run_scan(tuple(tickers), int(dte_min), int(dte_max), int(earn_days), int(window))
        except Exception as exc:
            st.error(f"Scan failed: {exc}. Yahoo may be busy. Try again in a few minutes.")
            st.stop()

scan = st.session_state.get("scan")
if scan is None:
    st.info("Set your tickers and rules on the left, then press Run scan.")
    st.stop()

age = dt.datetime.now() - scan["time"]
st.write(f"Data as of {scan['time']:%b %d %I:%M %p} server time (UTC), {len(scan['chain']):,} contracts.")
if age > dt.timedelta(hours=6):
    st.markdown('<div class="warn">This scan is more than 6 hours old. Prices have moved. Rescan before trading.</div>',
                unsafe_allow_html=True)
for w in scan["warnings"]:
    box(w, "warn")

chain, profiles, history = scan["chain"], scan["profiles"], scan["history"]
owned_set = {t.strip().upper() for t in owned.replace(",", " ").split()}

tabs = st.tabs(["Premium selling", "Directional", "Earnings", "Market overview", "Journal", "How it works"])


def jdf() -> pd.DataFrame:
    if "journal" not in st.session_state:
        st.session_state["journal"] = pd.DataFrame(columns=journal.COLS)
    return st.session_state["journal"]


def jadd(rec: dict):
    df = jdf()
    rec = {c: rec.get(c, "") for c in journal.COLS}
    rec["id"] = int(pd.to_numeric(df["id"]).max()) + 1 if len(df) else 1
    rec["opened"], rec["status"] = dt.date.today().isoformat(), "open"
    st.session_state["journal"] = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)


def log_form(df: pd.DataFrame, kind: str, entry_col: str):
    with st.expander("Log one of these trades to your journal"):
        n = st.number_input("Row number", 1, max(1, len(df)), 1, key=f"row_{kind}")
        thesis = st.text_input("Why are you taking it?", key=f"th_{kind}")
        qty = st.number_input("Contracts", 1, 1000, 1, key=f"q_{kind}")
        if st.button("Log trade", key=f"log_{kind}"):
            r = df.iloc[int(n) - 1]
            jadd({"ticker": r.ticker, "strategy": r.get("strategy", r.get("verdict", "")), "legs": r.legs,
                         "expiration": r.expiration, "contracts": qty, "entry": r.get(entry_col, ""),
                         "max_loss": r.get("max_loss", ""), "pop_at_entry": r.get("prob_profit", ""),
                         "ev_at_entry": r.get("ev", ""), "thesis": thesis})
            st.success("Logged. See the Journal tab. Download it there before you leave, the site does not store it.")


# ---------------- Premium ----------------
with tabs[0]:
    st.header("Premium selling")
    inc = st.checkbox("Include trades with earnings before expiration (higher risk)", False)
    only_pos = st.checkbox("Only show positive expected value", True)
    df = strategies.premium_candidates(chain, profiles, rules, include_earnings=inc, owned=owned_set)
    if df.empty:
        st.write("No trades pass your rules. Loosen liquidity or delta settings, or scan more tickers.")
    else:
        if only_pos:
            df = df[df.ev > 0]
        strat = st.multiselect("Strategies", sorted(df.strategy.unique()), sorted(df.strategy.unique()))
        df = df[df.strategy.isin(strat)].reset_index(drop=True)
        if df.empty:
            st.write("Nothing with positive expected value right now. That is a valid answer. Do not force a trade.")
        else:
            top = df.iloc[0]
            text = narrate.premium(top)
            box(text, lead="<b>Top pick.</b> ")
            speak_button(" ".join(narrate.premium(r) for _, r in df.head(3).iterrows()), "prem")
            big_table(df[["ticker", "strategy", "legs", "expiration", "dte", "credit", "max_loss", "prob_profit",
                          "ev", "ev_per_risk", "iv_hv", "contracts", "flag"]],
                      {"credit": money, "max_loss": money, "ev": money, "prob_profit": pct,
                       "ev_per_risk": lambda x: pct(x, 1), "iv_hv": lambda x: f"{x:.2f}"}, 15)
            with st.expander("Full sortable table"):
                st.dataframe(df, width="stretch")
            log_form(df, "prem", "credit")

# ---------------- Directional ----------------
with tabs[1]:
    st.header("Directional buys")
    trend = pd.DataFrame([{"ticker": t, **{k: p.get(k) for k in ("trend", "last", "rsi", "ret20", "sma50", "sma200")}}
                          for t, p in profiles.items() if p])
    if not trend.empty:
        st.write("Trend filter: price above the 50 day average, 50 above 200, RSI 50 to 72 and positive 20 day return "
                 "is bullish. The mirror image is bearish. Everything else is skipped.")
        big_table(trend, {"last": lambda x: f"{x:,.2f}", "rsi": lambda x: f"{x:.0f}", "ret20": lambda x: pct(x, 1),
                          "sma50": lambda x: f"{x:,.2f}", "sma200": lambda x: f"{x:,.2f}"}, 50)
    inc_d = st.checkbox("Include trades with earnings before expiration", False, key="incd")
    dd = strategies.directional_candidates(chain, profiles, rules, include_earnings=inc_d)
    if dd.empty:
        st.write("No trending names with liquid options right now.")
    else:
        st.markdown('<div class="warn">Expected value here assumes you have no edge on direction, so it is '
                    'usually negative. The trend filter is your bet. Rank shows the least negative trades and the '
                    'cheapest breakevens. Size small.</div>', unsafe_allow_html=True)
        top = dd.iloc[0]
        box(narrate.directional(top), lead="<b>Top pick.</b> ")
        speak_button(" ".join(narrate.directional(r) for _, r in dd.head(3).iterrows()), "dir")
        big_table(dd[["ticker", "trend", "strategy", "legs", "expiration", "debit", "max_gain", "breakeven",
                      "be_move_pct", "be_vs_expected", "prob_profit", "ev", "contracts", "flag"]],
                  {"debit": money, "max_gain": money, "ev": money, "prob_profit": pct, "breakeven": lambda x: f"{x:,.2f}",
                   "be_move_pct": lambda x: pct(x, 1), "be_vs_expected": lambda x: f"{x:.2f}"}, 15)
        with st.expander("Full sortable table"):
            st.dataframe(dd, width="stretch")
        log_form(dd, "dir", "debit")

# ---------------- Earnings ----------------
with tabs[2]:
    st.header("Earnings plays")
    st.write("Compares the move the options market is pricing (at-the-money straddle) with how much the stock "
             "actually moved after its last reports.")
    ed = strategies.earnings_candidates(chain, history, rules)
    if ed.empty:
        st.write(f"No scanned tickers report within {earn_days} days, or their post-earnings expiry is not in the data.")
    else:
        for _, r in ed[ed.verdict != "No trade"].head(3).iterrows():
            box(narrate.earnings(r))
        speak_button(" ".join(narrate.earnings(r) for _, r in ed.head(5).iterrows()), "earn")
        big_table(ed[["ticker", "earnings_date", "expiration", "implied_move", "hist_avg_move", "ratio", "beat_rate",
                      "reports", "verdict", "legs", "credit(+)/debit(-)", "max_loss", "contracts"]],
                  {"implied_move": lambda x: pct(x, 1), "hist_avg_move": lambda x: pct(x, 1),
                   "ratio": lambda x: "" if x != x else f"{x:.2f}", "beat_rate": pct,
                   "credit(+)/debit(-)": money, "max_loss": money}, 30)
        for t, mv in history.items():
            if mv:
                with st.expander(f"{t} past earnings moves"):
                    st.markdown(pd.DataFrame(mv).assign(move=lambda d: d.move.map(lambda x: f"{x:+.1%}")).to_html(index=False),
                                unsafe_allow_html=True)
        log_form(ed[ed.legs != ""].reset_index(drop=True) if (ed.legs != "").any() else ed, "earn", "credit(+)/debit(-)")

# ---------------- Overview ----------------
with tabs[3]:
    st.header("Market overview")
    rows = []
    for t, p in profiles.items():
        if not p:
            continue
        rows.append({"ticker": t, "price": p.get("last"), "trend": p.get("trend"), "atm_iv": p.get("atm_iv"),
                     "hv20": p.get("hv20"), "hv60": p.get("hv60"),
                     "iv_vs_hv": (p.get("atm_iv") or float("nan")) / p["hv_forecast"] if p.get("hv_forecast") else float("nan"),
                     "iv_rank": p.get("iv_rank"), "iv_history_days": p.get("iv_days", 0)})
    ov = pd.DataFrame(rows)
    if not ov.empty:
        ov = ov.sort_values("iv_vs_hv", ascending=False)
        st.write("IV vs HV above 1.2 means options look expensive versus how the stock has moved: favors selling. "
                 "Below 0.9 favors buying. IV rank fills in after about 20 days of scans.")
        big_table(ov, {"price": lambda x: f"{x:,.2f}", "atm_iv": pct, "hv20": pct, "hv60": pct,
                       "iv_vs_hv": lambda x: "" if x != x else f"{x:.2f}",
                       "iv_rank": lambda x: "building" if x != x else f"{x:.0f}"}, 60)
    with st.expander("Raw option chain"):
        st.dataframe(chain, width="stretch")
        st.download_button("Download chain as CSV", chain.to_csv(index=False), "chain.csv")

# ---------------- Journal ----------------
with tabs[4]:
    st.header("Trade journal")
    box("Your journal lives only in this browser tab. Download it before you close the page. "
        "Next time, upload the file to pick up where you left off.", "warn")
    up = st.file_uploader("Upload a saved journal (CSV)", type="csv")
    if up is not None and st.session_state.get("journal_loaded") != up.name:
        try:
            loaded = pd.read_csv(up)
            st.session_state["journal"] = loaded.reindex(columns=journal.COLS)
            st.session_state["journal_loaded"] = up.name
            st.success(f"Loaded {len(loaded)} trades.")
        except Exception as exc:
            st.error(f"Could not read that file: {exc}")
    j = jdf()
    s_ = journal.stats(j)
    if s_:
        st.write(f"Closed trades {s_['closed']}. Win rate {s_['win_rate']:.0%}. Total P&L \\${s_['total_pnl']:,.0f}. "
                 f"Average win \\${s_['avg_win']:,.0f}. Average loss \\${s_['avg_loss']:,.0f}.")
    if j.empty:
        st.write("No trades logged yet. Use the 'Log one of these trades' box under any scanner.")
    else:
        st.markdown(f'<div style="overflow-x:auto">{j.to_html(index=False)}</div>', unsafe_allow_html=True)
        opens = j[j.status == "open"]
        if not opens.empty:
            with st.expander("Close a trade"):
                tid = st.selectbox("Trade id", opens.id.tolist())
                ex = st.number_input("Exit value per contract ($)", value=0.0)
                pnl = st.number_input("Total P&L ($)", value=0.0)
                lesson = st.text_input("What did you learn?")
                if st.button("Close trade"):
                    m = j["id"] == tid
                    j.loc[m, ["status", "closed", "exit", "pnl", "lesson"]] = [
                        "closed", dt.date.today().isoformat(), ex, pnl, lesson]
                    st.session_state["journal"] = j
                    st.rerun()
        st.download_button("Download journal CSV", j.to_csv(index=False), "options_journal.csv", type="primary")

# ---------------- Methodology ----------------
with tabs[5]:
    st.header("How it works")
    st.markdown("""
**Data.** Option chains, price history and earnings dates all come from Yahoo Finance for free.
Quotes are delayed. Treat every number as a starting point and confirm in your broker before you trade.

**Fills.** The scanner never assumes you fill at mid. The slippage slider moves your fill toward the bad side
of the bid/ask. At 0.5 you give up half the distance from mid to the bid or ask on every leg.

**Expected value.** Each trade's payoff is averaged over a lognormal distribution of where the stock could
finish, using your realized volatility forecast and no directional drift. If you priced with implied volatility,
every trade would look fair. The edge comes only from implied volatility being higher or lower than what the
stock actually does.

**Probability of profit.** The share of that distribution where the trade makes money at expiration.
It ignores early management. Closing at 50% of max profit changes the real numbers.

**Earnings.** Trades whose expiry falls after an earnings report are hidden from the first two tabs by default.
Realized volatility does not capture an earnings gap, so those trades look safer than they are.
The Earnings tab handles them with a separate test: implied move versus past moves.

**Sizing.** Contracts = account size times your max loss percent, divided by max loss per contract.
Zero means the trade is too big for your rules. Respect the zero.

**Known limits.**
1. Greeks and probabilities use Black-Scholes. American early exercise and dividends are ignored.
2. IV rank needs about 20 days of saved scans and resets when the site restarts. Use IV / HV instead.
3. A positive EV estimate is only as good as the volatility forecast. Volatility regimes change.
4. Past earnings moves are a small sample, often 8 to 12 reports.
""")
