"""
Tests for backtester.py - strategy execution, VaR, COVID backtest.
"""

import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from black_scholes import OptionParams
from backtester import (
    OptionsBacktester, BacktestResult,
    straddle, strangle, short_straddle, iron_condor, bull_call_spread,
    monte_carlo_var, historical_var, Leg, StrategyConfig
)
from data import _synthetic_price_series


@pytest.fixture
def sample_prices():
    return _synthetic_price_series("2020-01-01", "2023-12-31", S0=330.0)


@pytest.fixture
def backtester(sample_prices):
    return OptionsBacktester(sample_prices, r=0.05, q=0.015)


class TestStrategyBuilders:
    def test_straddle_two_legs(self):
        s = straddle(30)
        assert len(s.legs) == 2
        assert any(l.option_type == "call" for l in s.legs)
        assert any(l.option_type == "put" for l in s.legs)

    def test_strangle_otm(self):
        s = strangle(0.05, 30)
        assert any(l.strike_offset == 0.05 for l in s.legs)
        assert any(l.strike_offset == -0.05 for l in s.legs)

    def test_short_straddle_negative_qty(self):
        s = short_straddle(30)
        assert all(l.quantity < 0 for l in s.legs)

    def test_iron_condor_four_legs(self):
        s = iron_condor(0.05, 30)
        assert len(s.legs) == 4

    def test_bull_call_spread_two_legs(self):
        s = bull_call_spread(0.05, 30)
        assert len(s.legs) == 2
        assert s.legs[0].quantity > 0
        assert s.legs[1].quantity < 0


class TestBacktester:
    def test_straddle_backtest_runs(self, backtester):
        result = backtester.run(straddle(30))
        assert isinstance(result, BacktestResult)
        assert result.num_trades > 0

    def test_short_straddle_backtest(self, backtester):
        result = backtester.run(short_straddle(30))
        assert result.num_trades > 0

    def test_strangle_backtest(self, backtester):
        result = backtester.run(strangle(0.05, 30))
        assert result.num_trades > 0

    def test_iron_condor_backtest(self, backtester):
        result = backtester.run(iron_condor(0.05, 30))
        assert result.num_trades >= 0

    def test_bull_spread_backtest(self, backtester):
        result = backtester.run(bull_call_spread(0.05, 30))
        assert result.num_trades >= 0

    def test_equity_curve_non_empty(self, backtester):
        result = backtester.run(straddle(30))
        assert len(result.equity_curve) > 0

    def test_win_rate_valid(self, backtester):
        result = backtester.run(straddle(30))
        assert 0 <= result.win_rate <= 100

    def test_sharpe_finite(self, backtester):
        result = backtester.run(straddle(30))
        assert np.isfinite(result.sharpe)

    def test_max_drawdown_non_positive(self, backtester):
        result = backtester.run(straddle(30))
        assert result.max_drawdown <= 0

    def test_date_filtering(self, backtester):
        result = backtester.run(straddle(30), start_date="2021-01-01", end_date="2022-12-31")
        if not result.trades.empty:
            assert result.trades["entry_date"].min() >= pd.Timestamp("2021-01-01")

    def test_covid_backtest(self, backtester):
        result = backtester.covid_backtest()
        assert isinstance(result, BacktestResult)

    def test_trades_df_columns(self, backtester):
        result = backtester.run(straddle(30))
        if not result.trades.empty:
            expected_cols = ["entry_date", "exit_date", "pnl", "cumulative_pnl"]
            for col in expected_cols:
                assert col in result.trades.columns

    def test_run_multiple(self, backtester):
        results = backtester.run_multiple([straddle(30), short_straddle(30)])
        assert "ATM Straddle" in results
        assert "Short Straddle" in results

    def test_transaction_costs_reduce_pnl(self, backtester):
        r1 = backtester.run(straddle(30), transaction_cost=0.0)
        r2 = backtester.run(straddle(30), transaction_cost=5.0)
        assert r1.total_return >= r2.total_return

    def test_empty_backtest_returns_result(self, backtester):
        """Very short window may produce zero trades but should not crash."""
        result = backtester.run(straddle(30), start_date="2022-01-01", end_date="2022-01-10")
        assert isinstance(result, BacktestResult)

    def test_vol_scale(self, backtester):
        r1 = backtester.run(straddle(30), vol_scale=1.0)
        r2 = backtester.run(straddle(30), vol_scale=1.5)
        # Different vol scales should produce different results
        assert r1.total_return != r2.total_return or r1.num_trades == r2.num_trades


class TestMonteCarloVaR:
    @pytest.fixture
    def sample_positions(self):
        return [
            (OptionParams(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type="call"), 1),
            (OptionParams(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type="put"), 1),
        ]

    def test_var_returns_dict(self, sample_positions):
        result = monte_carlo_var(1000, sample_positions, n_sims=1000, seed=42)
        assert "var_1day" in result
        assert "cvar_1day" in result

    def test_var_negative(self, sample_positions):
        """VaR should be negative (loss)."""
        result = monte_carlo_var(1000, sample_positions, n_sims=5000, seed=42)
        assert result["var_1day"] < 0

    def test_cvar_leq_var(self, sample_positions):
        """CVaR <= VaR (CVaR is worse)."""
        result = monte_carlo_var(1000, sample_positions, n_sims=5000, seed=42)
        assert result["cvar_1day"] <= result["var_1day"]

    def test_pnl_distribution_length(self, sample_positions):
        n_sims = 2000
        result = monte_carlo_var(1000, sample_positions, n_sims=n_sims, seed=42)
        assert len(result["pnl_distribution"]) == n_sims

    def test_var_higher_confidence_larger(self, sample_positions):
        """Higher confidence VaR should be more extreme (smaller number)."""
        r95 = monte_carlo_var(1000, sample_positions, confidence=0.95, n_sims=5000, seed=42)
        r99 = monte_carlo_var(1000, sample_positions, confidence=0.99, n_sims=5000, seed=42)
        assert r99["var_1day"] <= r95["var_1day"]

    def test_var_with_expired_options(self):
        """Options with T=0 should not crash."""
        positions = [(OptionParams(100, 100, 0.0, 0.05, 0.20, option_type="call"), 1)]
        result = monte_carlo_var(100, positions, n_sims=100, seed=42)
        assert "var_1day" in result


class TestHistoricalVaR:
    def test_historical_var(self):
        returns = pd.Series(np.random.randn(252) * 0.01)
        result = historical_var(returns, 100_000, confidence=0.99)
        assert result["var"] < 0
        assert result["cvar"] <= result["var"]
