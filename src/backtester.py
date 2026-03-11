"""
Options Backtester
Supports: Straddle, Strangle, Bull/Bear Spread, Iron Condor, Covered Call
COVID crash (Feb-Apr 2020) and 2020-2026 full backtest included.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Callable
from black_scholes import bs_price, OptionParams
from greeks import compute_greeks


@dataclass
class Leg:
    """A single option leg in a strategy."""
    option_type: str      # "call" or "put"
    strike_offset: float  # relative to ATM spot (e.g., 0.0 = ATM, 0.05 = 5% OTM)
    expiry_days: int      # DTE at entry
    quantity: int         # positive = long, negative = short
    strike_abs: float = 0.0  # computed at entry


@dataclass
class StrategyConfig:
    name: str
    legs: List[Leg]
    entry_dte: int = 30          # days before expiry to enter
    exit_dte: int = 5            # days before expiry to exit (0 = hold to expiry)
    rebalance_delta: bool = False
    delta_hedge_threshold: float = 0.05


@dataclass
class BacktestResult:
    strategy: str
    trades: pd.DataFrame
    equity_curve: pd.Series
    total_return: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    avg_pnl: float
    avg_win: float
    avg_loss: float
    num_trades: int
    greeks_history: Optional[pd.DataFrame] = None


# ---------------------------------------------------------------------------
# Pre-defined Strategies
# ---------------------------------------------------------------------------

def straddle(expiry_days: int = 30, quantity: int = 1) -> StrategyConfig:
    """ATM straddle: long call + long put at same strike."""
    return StrategyConfig(
        name="ATM Straddle",
        legs=[
            Leg("call", 0.0, expiry_days, quantity),
            Leg("put", 0.0, expiry_days, quantity),
        ],
        entry_dte=expiry_days,
        exit_dte=5,
    )


def strangle(otm_pct: float = 0.05, expiry_days: int = 30,
             quantity: int = 1) -> StrategyConfig:
    """OTM strangle: long OTM call + long OTM put."""
    return StrategyConfig(
        name=f"{otm_pct*100:.0f}% OTM Strangle",
        legs=[
            Leg("call", otm_pct, expiry_days, quantity),
            Leg("put", -otm_pct, expiry_days, quantity),
        ],
        entry_dte=expiry_days,
        exit_dte=5,
    )


def short_straddle(expiry_days: int = 30) -> StrategyConfig:
    """Short ATM straddle (vega seller)."""
    return StrategyConfig(
        name="Short Straddle",
        legs=[
            Leg("call", 0.0, expiry_days, -1),
            Leg("put", 0.0, expiry_days, -1),
        ],
        entry_dte=expiry_days,
        exit_dte=5,
    )


def iron_condor(wing_pct: float = 0.05, expiry_days: int = 30) -> StrategyConfig:
    """Iron condor: short OTM strangle + long further OTM strangle."""
    return StrategyConfig(
        name="Iron Condor",
        legs=[
            Leg("call", wing_pct, expiry_days, -1),       # short call
            Leg("call", wing_pct * 2, expiry_days, 1),    # long call wing
            Leg("put", -wing_pct, expiry_days, -1),        # short put
            Leg("put", -wing_pct * 2, expiry_days, 1),     # long put wing
        ],
        entry_dte=expiry_days,
        exit_dte=5,
    )


def bull_call_spread(width_pct: float = 0.05, expiry_days: int = 30) -> StrategyConfig:
    """Bull call spread: long ATM call + short OTM call."""
    return StrategyConfig(
        name="Bull Call Spread",
        legs=[
            Leg("call", 0.0, expiry_days, 1),
            Leg("call", width_pct, expiry_days, -1),
        ],
        entry_dte=expiry_days,
        exit_dte=5,
    )


def covered_call(otm_pct: float = 0.02, expiry_days: int = 30) -> StrategyConfig:
    """Covered call: assumed long 100 shares + short call."""
    return StrategyConfig(
        name="Covered Call",
        legs=[Leg("call", otm_pct, expiry_days, -1)],
        entry_dte=expiry_days,
        exit_dte=5,
    )


# ---------------------------------------------------------------------------
# Core Backtester
# ---------------------------------------------------------------------------

class OptionsBacktester:
    def __init__(
        self,
        price_series: pd.Series,
        vol_series: Optional[pd.Series] = None,
        r: float = 0.05,
        q: float = 0.015,
        contract_multiplier: int = 100,
    ):
        """
        price_series: pd.Series of daily closing prices (DatetimeIndex)
        vol_series: annualised historical vol (30-day rolling if None)
        r: risk-free rate
        q: dividend yield
        """
        self.prices = price_series.dropna()
        self.r = r
        self.q = q
        self.multiplier = contract_multiplier

        if vol_series is not None:
            self.vols = vol_series.reindex(self.prices.index).ffill()
        else:
            log_ret = np.log(self.prices / self.prices.shift(1))
            self.vols = log_ret.rolling(21).std() * np.sqrt(252)
            self.vols = self.vols.ffill().bfill()

    def _option_price(self, S: float, K: float, T: float, sigma: float,
                      option_type: str) -> float:
        """Price a single option using Black-Scholes."""
        if T <= 0:
            if option_type == "call":
                return max(S - K, 0.0)
            return max(K - S, 0.0)
        p = OptionParams(S=S, K=K, T=T, r=self.r, sigma=sigma,
                         q=self.q, option_type=option_type)
        return bs_price(p)

    def _strategy_value(
        self, legs: List[Leg], S: float, T_rem: float, sigma: float
    ) -> float:
        """Total mark-to-market value of all legs."""
        total = 0.0
        for leg in legs:
            K = leg.strike_abs
            price = self._option_price(S, K, max(T_rem, 0), sigma, leg.option_type)
            total += leg.quantity * price * self.multiplier
        return total

    def run(
        self,
        strategy: StrategyConfig,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        vol_scale: float = 1.0,   # scale vol (e.g., 1.2 for VRP)
        transaction_cost: float = 0.50,  # $ per contract per leg
        verbose: bool = False,
    ) -> BacktestResult:
        """
        Run backtest of strategy over price history.
        Enters a new position every entry_dte days.
        """
        prices = self.prices
        vols = self.vols * vol_scale

        if start_date:
            prices = prices[prices.index >= start_date]
            vols = vols[vols.index >= start_date]
        if end_date:
            prices = prices[prices.index <= end_date]
            vols = vols[vols.index <= end_date]

        dates = prices.index
        entry_dte = strategy.entry_dte
        exit_dte = strategy.exit_dte

        trades = []
        equity_curve_vals = []
        running_pnl = 0.0

        i = 0
        while i < len(dates) - exit_dte - 1:
            entry_date = dates[i]
            exit_idx = min(i + entry_dte - exit_dte, len(dates) - 1)
            exit_date = dates[exit_idx]

            S_entry = prices.iloc[i]
            S_exit = prices.iloc[exit_idx]
            sigma_entry = vols.iloc[i] if not np.isnan(vols.iloc[i]) else 0.20
            sigma_exit = vols.iloc[exit_idx] if not np.isnan(vols.iloc[exit_idx]) else 0.20

            # Set absolute strikes
            legs = []
            for leg_template in strategy.legs:
                leg = Leg(
                    option_type=leg_template.option_type,
                    strike_offset=leg_template.strike_offset,
                    expiry_days=leg_template.expiry_days,
                    quantity=leg_template.quantity,
                )
                leg.strike_abs = S_entry * (1 + leg.strike_offset)
                legs.append(leg)

            T_entry = entry_dte / 252.0
            T_exit = exit_dte / 252.0

            # Entry value (debit/credit)
            entry_value = self._strategy_value(legs, S_entry, T_entry, sigma_entry)
            exit_value = self._strategy_value(legs, S_exit, T_exit, sigma_exit)

            # PnL = exit - entry (accounting for long/short)
            n_legs = sum(abs(leg.quantity) for leg in legs)
            costs = transaction_cost * n_legs * 2  # round trip
            trade_pnl = exit_value - entry_value - costs

            running_pnl += trade_pnl
            equity_curve_vals.append((exit_date, running_pnl))

            # Greeks at entry
            entry_greeks = self._portfolio_greeks(legs, S_entry, T_entry, sigma_entry)

            trades.append({
                "entry_date": entry_date,
                "exit_date": exit_date,
                "S_entry": S_entry,
                "S_exit": S_exit,
                "sigma_entry": sigma_entry,
                "sigma_exit": sigma_exit,
                "move_pct": (S_exit / S_entry - 1) * 100,
                "entry_value": entry_value,
                "exit_value": exit_value,
                "pnl": trade_pnl,
                "cumulative_pnl": running_pnl,
                "delta_entry": entry_greeks.get("delta", 0),
                "gamma_entry": entry_greeks.get("gamma", 0),
                "vega_entry": entry_greeks.get("vega", 0),
                "theta_entry": entry_greeks.get("theta", 0),
            })

            i += entry_dte

        trades_df = pd.DataFrame(trades)
        if trades_df.empty:
            return BacktestResult(
                strategy=strategy.name, trades=trades_df,
                equity_curve=pd.Series(dtype=float),
                total_return=0, sharpe=0, max_drawdown=0,
                win_rate=0, avg_pnl=0, avg_win=0, avg_loss=0, num_trades=0,
            )

        equity_series = pd.Series(
            [v for _, v in equity_curve_vals],
            index=[d for d, _ in equity_curve_vals],
        )

        pnls = trades_df["pnl"]
        winners = pnls[pnls > 0]
        losers = pnls[pnls <= 0]

        # Sharpe (annualised, 252 trading days)
        if len(pnls) > 1 and pnls.std() > 0:
            # Approximate annualisation by trade frequency
            trades_per_year = 252 / max(entry_dte, 1)
            sharpe = (pnls.mean() / pnls.std()) * np.sqrt(trades_per_year)
        else:
            sharpe = 0.0

        # Max drawdown
        cumulative = equity_series
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max)
        max_dd = float(drawdown.min())

        return BacktestResult(
            strategy=strategy.name,
            trades=trades_df,
            equity_curve=equity_series,
            total_return=float(running_pnl),
            sharpe=float(sharpe),
            max_drawdown=max_dd,
            win_rate=float(len(winners) / len(pnls) * 100) if len(pnls) > 0 else 0,
            avg_pnl=float(pnls.mean()),
            avg_win=float(winners.mean()) if len(winners) > 0 else 0,
            avg_loss=float(losers.mean()) if len(losers) > 0 else 0,
            num_trades=len(pnls),
        )

    def _portfolio_greeks(self, legs: List[Leg], S: float, T: float,
                          sigma: float) -> Dict[str, float]:
        """Aggregate portfolio Greeks for all legs."""
        total = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
        for leg in legs:
            if T <= 0:
                continue
            p = OptionParams(S=S, K=leg.strike_abs, T=T, r=self.r,
                             sigma=sigma, q=self.q, option_type=leg.option_type)
            g = compute_greeks(p)
            for key in total:
                total[key] += getattr(g, key) * leg.quantity * self.multiplier
        return total

    def run_multiple(
        self,
        strategies: List[StrategyConfig],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, BacktestResult]:
        """Run multiple strategies and return comparative results."""
        return {
            s.name: self.run(s, start_date=start_date, end_date=end_date)
            for s in strategies
        }

    def covid_backtest(self, strategy: Optional[StrategyConfig] = None) -> BacktestResult:
        """Quick COVID crash backtest (Feb 19 – Apr 30, 2020)."""
        strat = strategy or straddle(30)
        return self.run(strat, start_date="2020-01-01", end_date="2020-06-30")


# ---------------------------------------------------------------------------
# VaR & Monte Carlo Risk
# ---------------------------------------------------------------------------

def monte_carlo_var(
    portfolio_value: float,
    positions: List[Tuple[OptionParams, int]],
    horizon_days: int = 1,
    n_sims: int = 10_000,
    confidence: float = 0.99,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Monte Carlo VaR for options portfolio.
    Simulates GBM paths and reprices portfolio at each scenario.
    """
    np.random.seed(seed)

    # Reference portfolio value
    ref_value = sum(
        bs_price(p) * qty * 100
        for p, qty in positions
        if p.T > 0
    )

    # Draw scenarios
    dt = horizon_days / 252.0
    pnls = np.zeros(n_sims)

    for p, qty in positions:
        if p.T <= 0:
            continue
        sigma = p.sigma
        r, q = p.r, p.q
        S0 = p.S

        # GBM spot shock
        z = np.random.standard_normal(n_sims)
        S_shocked = S0 * np.exp((r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)

        # Vol shock (correlated)
        z_vol = -0.7 * z + np.sqrt(1 - 0.49) * np.random.standard_normal(n_sims)
        sigma_shocked = np.maximum(sigma * np.exp(0.3 * np.sqrt(dt) * z_vol - 0.045 * dt), 0.01)

        T_new = max(p.T - dt, 0)
        new_prices = np.where(
            T_new > 0,
            np.array([
                bs_price(OptionParams(S=s, K=p.K, T=T_new, r=r,
                                     sigma=sig, q=q, option_type=p.option_type))
                for s, sig in zip(S_shocked, sigma_shocked)
            ]),
            np.maximum(S_shocked - p.K, 0) if p.option_type == "call"
            else np.maximum(p.K - S_shocked, 0),
        )

        ref_price = bs_price(p)
        pnls += (new_prices - ref_price) * qty * 100

    var_1day = float(np.percentile(pnls, (1 - confidence) * 100))
    cvar = float(pnls[pnls <= var_1day].mean()) if np.any(pnls <= var_1day) else var_1day

    return {
        "var_1day": var_1day,
        "cvar_1day": cvar,
        "var_scaled": var_1day * np.sqrt(horizon_days),
        "mean_pnl": float(pnls.mean()),
        "std_pnl": float(pnls.std()),
        "worst_case": float(pnls.min()),
        "best_case": float(pnls.max()),
        "pnl_distribution": pnls,
    }


def historical_var(
    returns: pd.Series,
    portfolio_value: float,
    confidence: float = 0.99,
    horizon_days: int = 1,
) -> Dict[str, float]:
    """Historical simulation VaR from return series."""
    scaled = returns * np.sqrt(horizon_days) * portfolio_value
    var = float(np.percentile(scaled, (1 - confidence) * 100))
    cvar = float(scaled[scaled <= var].mean())
    return {
        "var": var,
        "cvar": cvar,
        "confidence": confidence,
        "horizon_days": horizon_days,
    }
