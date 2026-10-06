"""Trade journal stored as a CSV you can open in Excel."""
from __future__ import annotations

import datetime as dt

import pandas as pd

from sc_config import DATA_DIR

PATH = DATA_DIR / "journal.csv"
COLS = ["id", "opened", "ticker", "strategy", "legs", "expiration", "contracts", "entry", "max_loss",
        "pop_at_entry", "ev_at_entry", "thesis", "status", "closed", "exit", "pnl", "lesson"]


def load() -> pd.DataFrame:
    if PATH.exists():
        return pd.read_csv(PATH)
    return pd.DataFrame(columns=COLS)


def add(rec: dict) -> None:
    df = load()
    rec = {c: rec.get(c, "") for c in COLS}
    rec["id"] = int(df["id"].max()) + 1 if len(df) else 1
    rec["opened"] = dt.date.today().isoformat()
    rec["status"] = "open"
    pd.concat([df, pd.DataFrame([rec])], ignore_index=True).to_csv(PATH, index=False)


def close(trade_id: int, exit_value: float, pnl: float, lesson: str) -> None:
    df = load()
    m = df["id"] == trade_id
    df.loc[m, ["status", "closed", "exit", "pnl", "lesson"]] = ["closed", dt.date.today().isoformat(),
                                                                 exit_value, pnl, lesson]
    df.to_csv(PATH, index=False)


def stats(df: pd.DataFrame) -> dict:
    c = df[df.status == "closed"].copy()
    if c.empty:
        return {}
    c["pnl"] = pd.to_numeric(c["pnl"], errors="coerce")
    wins = c[c.pnl > 0]
    return {"closed": len(c), "win_rate": len(wins) / len(c), "total_pnl": c.pnl.sum(),
            "avg_win": wins.pnl.mean() if len(wins) else 0.0,
            "avg_loss": c[c.pnl <= 0].pnl.mean() if (c.pnl <= 0).any() else 0.0}
