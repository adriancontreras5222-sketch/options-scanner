"""Black-Scholes pricing, greeks, implied vol, and payoff statistics under a lognormal model.

All functions are vectorised with numpy where it matters. T is in years. sigma is annual vol.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _d1d2(S, K, T, r, sigma, q=0.0):
    S, K, T, sigma = map(np.asarray, (S, K, T, sigma))
    T = np.maximum(T, 1e-8)
    sigma = np.maximum(sigma, 1e-8)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def price(S, K, T, r, sigma, kind="call", q=0.0):
    d1, d2 = _d1d2(S, K, T, r, sigma, q)
    if kind == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def delta(S, K, T, r, sigma, kind="call", q=0.0):
    d1, _ = _d1d2(S, K, T, r, sigma, q)
    return np.exp(-q * T) * (norm.cdf(d1) if kind == "call" else norm.cdf(d1) - 1)


def gamma(S, K, T, r, sigma, q=0.0):
    d1, _ = _d1d2(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(np.maximum(T, 1e-8)))


def vega(S, K, T, r, sigma, q=0.0):
    """Per 1 vol point (0.01)."""
    d1, _ = _d1d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(np.maximum(T, 1e-8)) / 100


def theta(S, K, T, r, sigma, kind="call", q=0.0):
    """Per calendar day."""
    d1, d2 = _d1d2(S, K, T, r, sigma, q)
    T = np.maximum(T, 1e-8)
    first = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    if kind == "call":
        val = first - r * K * np.exp(-r * T) * norm.cdf(d2) + q * S * np.exp(-q * T) * norm.cdf(d1)
    else:
        val = first + r * K * np.exp(-r * T) * norm.cdf(-d2) - q * S * np.exp(-q * T) * norm.cdf(-d1)
    return val / 365


def implied_vol(px, S, K, T, r, kind="call", q=0.0) -> float:
    """Implied vol from a premium. Returns nan when the premium is outside no-arbitrage bounds."""
    if not (px and px > 0 and S > 0 and K > 0 and T > 0):
        return float("nan")
    intrinsic = max(0.0, (S - K) if kind == "call" else (K - S))
    if px <= intrinsic * math.exp(-r * T) + 1e-6:
        return float("nan")
    f = lambda s: float(price(S, K, T, r, s, kind, q)) - px  # noqa: E731
    try:
        return brentq(f, 1e-4, 6.0, maxiter=200)
    except ValueError:
        return float("nan")


# ---------- distribution of the stock at expiry ----------

def terminal_grid(S: float, T: float, sigma: float, mu: float = 0.0, n: int = 4001):
    """Grid of terminal prices and probability weights under a lognormal with annual drift mu.

    mu=0 means no directional edge assumed (the median drifts slightly down by -0.5*sigma^2*T,
    which is the honest real-world assumption without a forecast).
    """
    T = max(T, 1e-6)
    sd = sigma * math.sqrt(T)
    m = math.log(S) + (mu - 0.5 * sigma**2) * T
    z = np.linspace(-7, 7, n)
    ST = np.exp(m + sd * z)
    w = norm.pdf(z)
    w /= w.sum()
    return ST, w


def payoff_stats(payoff: Callable[[np.ndarray], np.ndarray], S: float, T: float, sigma: float, mu: float = 0.0):
    """Expected P&L and probability of profit for a payoff function of the terminal price.

    payoff must return P&L per share INCLUDING the premium paid/received.
    """
    ST, w = terminal_grid(S, T, sigma, mu)
    pnl = payoff(ST)
    return float((pnl * w).sum()), float(w[pnl > 0].sum())


def prob_above(S: float, K: float, T: float, sigma: float, mu: float = 0.0) -> float:
    T = max(T, 1e-6)
    z = (math.log(S / K) + (mu - 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return float(norm.cdf(z))
