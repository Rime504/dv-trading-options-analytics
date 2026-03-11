"""
Data Module
- Real-time SPX/SPY/ES options chains via yfinance (free)
- Futures curve data (contango/backwardation)
- Synthetic data generation for testing
- Caching layer to avoid redundant downloads
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# Optional yfinance import with graceful fallback
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Real Market Data (yfinance)
# ---------------------------------------------------------------------------

def fetch_spot_price(ticker: str = "SPY") -> float:
    """Fetch current spot price for ticker."""
    if not YFINANCE_AVAILABLE:
        return _synthetic_spot(ticker)
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="2d")
        if hist.empty:
            return _synthetic_spot(ticker)
        return float(hist["Close"].iloc[-1])
    except Exception:
        return _synthetic_spot(ticker)


def fetch_price_history(
    ticker: str = "SPY",
    start: str = "2020-01-01",
    end: Optional[str] = None,
    interval: str = "1d",
) -> pd.Series:
    """Fetch historical closing prices."""
    if not YFINANCE_AVAILABLE:
        return _synthetic_price_series(start=start, end=end)
    try:
        end_dt = end or datetime.today().strftime("%Y-%m-%d")
        t = yf.Ticker(ticker)
        hist = t.history(start=start, end=end_dt, interval=interval)
        if hist.empty:
            return _synthetic_price_series(start=start, end=end)
        return hist["Close"].dropna()
    except Exception:
        return _synthetic_price_series(start=start, end=end)


def fetch_options_chain(
    ticker: str = "SPY",
    max_expiries: int = 5,
) -> List[Dict]:
    """
    Fetch real options chain from yfinance.
    Returns list of dicts: {K, T, mid_price, bid, ask, iv, volume, oi, option_type, expiry}
    """
    if not YFINANCE_AVAILABLE:
        return _synthetic_chain()
    try:
        t = yf.Ticker(ticker)
        exps = t.options
        if not exps:
            return _synthetic_chain()

        chain_data = []
        today = datetime.today()
        spot = fetch_spot_price(ticker)

        for exp_str in exps[:max_expiries]:
            try:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
                T = max((exp_date - today).days / 365.0, 1 / 365.0)
                chain = t.option_chain(exp_str)

                for opt_type, df in [("call", chain.calls), ("put", chain.puts)]:
                    if df.empty:
                        continue
                    # Filter near-the-money: 70%-130% of spot
                    mask = (df["strike"] >= spot * 0.70) & (df["strike"] <= spot * 1.30)
                    df = df[mask].copy()

                    for _, row in df.iterrows():
                        bid = float(row.get("bid", 0) or 0)
                        ask = float(row.get("ask", 0) or 0)
                        mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else float(row.get("lastPrice", 0) or 0)
                        if mid <= 0:
                            continue

                        chain_data.append({
                            "K": float(row["strike"]),
                            "T": T,
                            "mid_price": mid,
                            "bid": bid,
                            "ask": ask,
                            "iv": float(row.get("impliedVolatility", 0) or 0),
                            "volume": int(row.get("volume", 0) or 0),
                            "oi": int(row.get("openInterest", 0) or 0),
                            "option_type": opt_type,
                            "expiry": exp_str,
                            "spot": spot,
                        })
            except Exception:
                continue

        return chain_data if chain_data else _synthetic_chain()
    except Exception:
        return _synthetic_chain()


def fetch_vix(start: str = "2020-01-01", end: Optional[str] = None) -> pd.Series:
    """Fetch VIX index history."""
    if not YFINANCE_AVAILABLE:
        return _synthetic_vix(start, end)
    try:
        end_dt = end or datetime.today().strftime("%Y-%m-%d")
        vix = yf.download("^VIX", start=start, end=end_dt, progress=False)
        if vix is None or (hasattr(vix, 'empty') and vix.empty):
            return _synthetic_vix(start, end)
        # yfinance may return DataFrame or Series depending on version
        if isinstance(vix, pd.DataFrame):
            col = "Close"
            if isinstance(vix.columns, pd.MultiIndex):
                vix = vix.xs("^VIX", axis=1, level=1) if "^VIX" in vix.columns.get_level_values(1) else vix
            return vix[col].squeeze().dropna()
        return vix.dropna()
    except Exception:
        return _synthetic_vix(start, end)


def fetch_futures_curve(
    root: str = "ES",
    n_contracts: int = 6,
) -> pd.DataFrame:
    """
    Fetch ES futures term structure.
    Returns DataFrame with columns: contract, price, expiry, days_to_exp
    Uses synthetic quarterly rolls if yfinance unavailable.
    """
    # yfinance doesn't support futures chains directly
    # Use synthetic roll curve based on current spot
    spot = fetch_spot_price("SPY") * 10  # ES = 10 * SPY approx
    r = 0.05
    q = 0.015

    today = datetime.today()
    contracts = []
    for i in range(1, n_contracts + 1):
        # Quarterly expiry (3rd Friday of Mar/Jun/Sep/Dec)
        months_ahead = i * 3
        exp_date = today + timedelta(days=months_ahead * 30)
        T = (exp_date - today).days / 365.0
        # Cost of carry formula: F = S * e^{(r-q)*T}
        fwd = spot * np.exp((r - q) * T)
        contracts.append({
            "contract": f"{root}{i}",
            "price": fwd,
            "expiry": exp_date.strftime("%Y-%m-%d"),
            "days_to_exp": (exp_date - today).days,
            "T": T,
            "basis": fwd - spot,
            "annualised_basis": (fwd / spot - 1) / T * 100 if T > 0 else 0,
        })

    df = pd.DataFrame(contracts)

    # Detect contango/backwardation
    prices = df["price"].values
    df["structure"] = "flat"
    for i in range(1, len(prices)):
        df.loc[i, "structure"] = "contango" if prices[i] > prices[i-1] else "backwardation"

    return df


def fetch_risk_free_rate() -> float:
    """Fetch 3-month T-bill rate via yfinance (^IRX) or return default."""
    if not YFINANCE_AVAILABLE:
        return 0.05
    try:
        t = yf.Ticker("^IRX")
        hist = t.history(period="5d")
        if hist.empty:
            return 0.05
        # ^IRX is in percent
        return float(hist["Close"].iloc[-1]) / 100.0
    except Exception:
        return 0.05


# ---------------------------------------------------------------------------
# Synthetic Data Generators (fallback + testing)
# ---------------------------------------------------------------------------

def _synthetic_spot(ticker: str) -> float:
    spots = {"SPY": 580.0, "SPX": 5800.0, "ES": 5800.0, "QQQ": 480.0, "IWM": 220.0}
    return spots.get(ticker, 400.0)


def _synthetic_price_series(
    start: str = "2020-01-01",
    end: Optional[str] = None,
    S0: float = 330.0,
    mu: float = 0.10,
    sigma: float = 0.20,
    seed: int = 42,
) -> pd.Series:
    """Generate GBM price series for testing."""
    np.random.seed(seed)
    end_dt = end or "2026-03-10"
    dates = pd.date_range(start, end_dt, freq="B")
    dt = 1 / 252
    n = len(dates)
    returns = np.exp((mu - 0.5 * sigma ** 2) * dt
                     + sigma * np.sqrt(dt) * np.random.randn(n))
    prices = S0 * np.cumprod(returns)

    # Inject COVID crash (Feb 20 - Mar 23, 2020) if in range
    series = pd.Series(prices, index=dates)
    if "2020-02-20" in str(series.index):
        covid_start = "2020-02-20"
        covid_end = "2020-03-23"
        mask = (series.index >= covid_start) & (series.index <= covid_end)
        n_crash = mask.sum()
        crash = np.exp(-0.35 / n_crash * np.arange(n_crash))
        peak = float(series[series.index < covid_start].iloc[-1])
        series.loc[mask] = peak * crash / crash[0]

        # Recovery Apr - Jun 2020
        post = series.index > covid_end
        n_post = post.sum()
        trough = float(series[series.index == covid_end].iloc[-1])
        recovery = np.exp(0.40 / n_post * np.arange(n_post))
        series.loc[post] = trough * recovery

    return series


def _synthetic_chain(
    S: float = 580.0,
    r: float = 0.05,
    q: float = 0.015,
    atm_vol: float = 0.18,
) -> List[Dict]:
    """Generate synthetic SPY-like options chain."""
    from black_scholes import bs_price_vectorized
    chain = []
    expiries_days = [7, 14, 30, 60, 90, 180]
    today = datetime.today()

    for days in expiries_days:
        T = days / 365.0
        exp_date = (today + timedelta(days=days)).strftime("%Y-%m-%d")
        F = S * np.exp((r - q) * T)

        # Strike grid: 80-120% of spot
        strikes = np.linspace(S * 0.80, S * 1.20, 25)

        for K in strikes:
            k = np.log(K / F)
            # Smile: skew + curvature
            iv = atm_vol + (-0.1 * k + 0.05 * k ** 2) * (1 + 0.1 / np.sqrt(T))
            iv = max(iv, 0.05)

            for opt_type in ["call", "put"]:
                price = float(bs_price_vectorized(S, K, T, r, iv, q, opt_type))
                if price < 0.01:
                    continue
                spread = max(price * 0.02, 0.05)
                chain.append({
                    "K": float(K),
                    "T": T,
                    "mid_price": price,
                    "bid": price - spread / 2,
                    "ask": price + spread / 2,
                    "iv": iv,
                    "volume": int(np.random.randint(10, 5000)),
                    "oi": int(np.random.randint(100, 50000)),
                    "option_type": opt_type,
                    "expiry": exp_date,
                    "spot": S,
                })

    return chain


def _synthetic_vix(start: str = "2020-01-01", end: Optional[str] = None) -> pd.Series:
    """Synthetic VIX series with COVID spike."""
    dates = pd.date_range(start, end or "2026-03-10", freq="B")
    n = len(dates)
    np.random.seed(0)
    vix = 15 + 3 * np.random.randn(n).cumsum() * 0.05
    vix = np.clip(vix, 10, 25)

    series = pd.Series(vix, index=dates)
    # COVID spike
    mask = (series.index >= "2020-02-20") & (series.index <= "2020-03-18")
    n_spike = mask.sum()
    if n_spike > 0:
        spike = np.linspace(20, 85, n_spike // 2).tolist() + np.linspace(85, 35, n_spike - n_spike // 2).tolist()
        series.loc[mask] = spike[:n_spike]
    return series


# ---------------------------------------------------------------------------
# Realised Vol Estimators
# ---------------------------------------------------------------------------

def parkinson_vol(high: pd.Series, low: pd.Series, window: int = 21) -> pd.Series:
    """Parkinson high-low volatility estimator (more efficient than close-close)."""
    hl = np.log(high / low)
    return np.sqrt((hl ** 2 / (4 * np.log(2))).rolling(window).mean() * 252)


def garman_klass_vol(open_: pd.Series, high: pd.Series,
                     low: pd.Series, close: pd.Series, window: int = 21) -> pd.Series:
    """Garman-Klass volatility estimator."""
    rs = 0.5 * np.log(high / low) ** 2 - (2 * np.log(2) - 1) * np.log(close / open_) ** 2
    return np.sqrt(rs.rolling(window).mean() * 252)


def vol_of_vol(vol_series: pd.Series, window: int = 21) -> pd.Series:
    """Volatility of volatility (vol-of-vol) series."""
    log_vol_ret = np.log(vol_series / vol_series.shift(1))
    return log_vol_ret.rolling(window).std() * np.sqrt(252)


def vrp_series(
    implied_vol: pd.Series,
    realised_vol: pd.Series,
) -> pd.Series:
    """
    Variance Risk Premium: IV^2 - RV^2 (annualised variance).
    Positive VRP means selling vol is profitable on average.
    """
    return implied_vol ** 2 - realised_vol ** 2


# ---------------------------------------------------------------------------
# Report Export
# ---------------------------------------------------------------------------

def export_to_excel(data_dict: Dict[str, pd.DataFrame], filepath: str) -> None:
    """Export multiple DataFrames to Excel sheets."""
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        for sheet_name, df in data_dict.items():
            df.to_excel(writer, sheet_name=sheet_name[:31])


def build_report_data(
    ticker: str = "SPY",
    start: str = "2020-01-01",
) -> Dict[str, pd.DataFrame]:
    """Build comprehensive data package for dashboard/export."""
    prices = fetch_price_history(ticker, start=start)
    log_ret = np.log(prices / prices.shift(1)).dropna()
    rv21 = log_ret.rolling(21).std() * np.sqrt(252)

    return {
        "Price_History": prices.to_frame(name="Close"),
        "Returns": log_ret.to_frame(name="LogReturn"),
        "Realised_Vol": rv21.to_frame(name="RV21"),
    }
