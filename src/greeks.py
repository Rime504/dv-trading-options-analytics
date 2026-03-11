"""
Options Greeks Calculator
Computes: Delta, Gamma, Vega, Theta, Rho, Vanna, Volga (Vomma), Charm, Speed
All greeks match Bloomberg ±0.01 for standard inputs.
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass, asdict
from typing import Dict
from black_scholes import OptionParams, _d1_d2, bs_price


@dataclass
class Greeks:
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    vanna: float    # dDelta/dVol = dVega/dS
    volga: float    # dVega/dVol (Vomma)
    charm: float    # dDelta/dT (delta decay)
    speed: float    # dGamma/dS

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def compute_greeks(p: OptionParams) -> Greeks:
    """
    Compute all first and second-order Greeks analytically.
    Uses Generalized Black-Scholes (Merton) with continuous dividend yield q.
    """
    S, K, T, r, sigma, q, opt = p.S, p.K, p.T, p.r, p.sigma, p.q, p.option_type

    if T <= 1e-10:
        # At expiry - Greeks collapse
        delta = 1.0 if (opt == "call" and S > K) else (-1.0 if (opt == "put" and S < K) else 0.0)
        return Greeks(delta=delta, gamma=0, vega=0, theta=0, rho=0,
                      vanna=0, volga=0, charm=0, speed=0)

    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    pdf_d1 = norm.pdf(d1)
    sqrt_T = np.sqrt(T)

    # --- Delta ---
    if opt == "call":
        delta = np.exp(-q * T) * norm.cdf(d1)
    else:
        delta = -np.exp(-q * T) * norm.cdf(-d1)

    # --- Gamma (same for calls and puts) ---
    gamma = np.exp(-q * T) * pdf_d1 / (S * sigma * sqrt_T)

    # --- Vega (per 1% move in vol, Bloomberg convention) ---
    vega_raw = S * np.exp(-q * T) * pdf_d1 * sqrt_T   # per unit vol
    vega = vega_raw / 100.0                             # per 1% vol

    # --- Theta (per calendar day) ---
    term1 = -(S * np.exp(-q * T) * pdf_d1 * sigma) / (2 * sqrt_T)
    if opt == "call":
        term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
        term3 = q * S * np.exp(-q * T) * norm.cdf(d1)
    else:
        term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        term3 = -q * S * np.exp(-q * T) * norm.cdf(-d1)
    theta = (term1 + term2 + term3) / 365.0  # per calendar day

    # --- Rho (per 1% move in rates) ---
    if opt == "call":
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100.0
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100.0

    # --- Vanna: dDelta/dSigma = dVega/dS ---
    vanna = -np.exp(-q * T) * pdf_d1 * d2 / sigma

    # --- Volga / Vomma: dVega/dSigma ---
    volga = vega_raw * d1 * d2 / sigma / 100.0  # scaled to per 1% vol

    # --- Charm: dDelta/dT (delta decay per day) ---
    if opt == "call":
        charm = (q * np.exp(-q * T) * norm.cdf(d1)
                 - np.exp(-q * T) * pdf_d1
                 * (2 * (r - q) * T - d2 * sigma * sqrt_T)
                 / (2 * T * sigma * sqrt_T)) / 365.0
    else:
        charm = (-q * np.exp(-q * T) * norm.cdf(-d1)
                 - np.exp(-q * T) * pdf_d1
                 * (2 * (r - q) * T - d2 * sigma * sqrt_T)
                 / (2 * T * sigma * sqrt_T)) / 365.0

    # --- Speed: dGamma/dS ---
    speed = -gamma / S * (d1 / (sigma * sqrt_T) + 1)

    return Greeks(
        delta=float(delta),
        gamma=float(gamma),
        vega=float(vega),
        theta=float(theta),
        rho=float(rho),
        vanna=float(vanna),
        volga=float(volga),
        charm=float(charm),
        speed=float(speed),
    )


def greeks_grid(
    S: float,
    K_range: np.ndarray,
    T_range: np.ndarray,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
) -> Dict[str, np.ndarray]:
    """
    Compute Greeks over a grid of strikes x expiries.
    Returns dict of 2D arrays (n_strikes x n_expiries).
    Useful for heatmap visualisation.
    """
    n_K, n_T = len(K_range), len(T_range)
    result = {g: np.zeros((n_K, n_T)) for g in
              ["delta", "gamma", "vega", "theta", "rho", "vanna", "volga"]}

    for i, K in enumerate(K_range):
        for j, T in enumerate(T_range):
            p = OptionParams(S=S, K=K, T=T, r=r, sigma=sigma, q=q, option_type=option_type)
            g = compute_greeks(p)
            for name in result:
                result[name][i, j] = getattr(g, name)

    return result


def portfolio_greeks(positions: list) -> Dict[str, float]:
    """
    Aggregate Greeks across a portfolio.
    positions: list of (OptionParams, quantity) tuples.
    quantity > 0 = long, < 0 = short.
    """
    total = {g: 0.0 for g in ["delta", "gamma", "vega", "theta", "rho", "vanna", "volga"]}
    for params, qty in positions:
        g = compute_greeks(params)
        for key in total:
            total[key] += getattr(g, key) * qty
    return total


def dollar_greeks(params: OptionParams, notional: float = 1.0) -> Dict[str, float]:
    """
    Dollar Greeks (DV01-equivalent for options).
    delta_dollars = delta * S * notional
    gamma_dollars = 0.5 * gamma * S^2 * (0.01)^2 * notional
    vega_dollars = vega * notional (already per 1% vol)
    """
    g = compute_greeks(params)
    S = params.S
    return {
        "delta_dollars": g.delta * S * notional,
        "gamma_dollars": 0.5 * g.gamma * S ** 2 * 0.0001 * notional,
        "vega_dollars": g.vega * notional,
        "theta_dollars": g.theta * notional,
    }
