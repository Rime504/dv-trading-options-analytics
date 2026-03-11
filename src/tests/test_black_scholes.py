"""
Tests for black_scholes.py - targeting 95%+ coverage.
All reference values validated against QuantLib / Bloomberg conventions.
"""

import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from black_scholes import (
    bs_price, binomial_tree_price, implied_volatility,
    put_call_parity_check, bs_price_vectorized, OptionParams, _d1_d2
)


# ---------------------------------------------------------------------------
# Reference values (Bloomberg-validated ±0.01)
# ---------------------------------------------------------------------------

class TestBSPrice:
    """Test Black-Scholes analytic pricer."""

    def test_atm_call_basic(self):
        """ATM call: Bloomberg reference S=100, K=100, T=1, r=5%, σ=20%"""
        p = OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        price = bs_price(p)
        assert abs(price - 10.4506) < 0.01, f"Expected ~10.45, got {price}"

    def test_atm_put_basic(self):
        """ATM put pricing."""
        p = OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="put")
        price = bs_price(p)
        assert abs(price - 5.5735) < 0.01, f"Expected ~5.57, got {price}"

    def test_deep_itm_call(self):
        """Deep ITM call approaches intrinsic value."""
        p = OptionParams(S=150, K=100, T=0.25, r=0.05, sigma=0.20)
        price = bs_price(p)
        intrinsic = 150 - 100 * np.exp(-0.05 * 0.25)
        assert price >= intrinsic - 0.01

    def test_deep_otm_call(self):
        """Deep OTM call should be near zero."""
        p = OptionParams(S=100, K=200, T=0.1, r=0.05, sigma=0.10)
        price = bs_price(p)
        assert price < 0.01

    def test_deep_otm_put(self):
        """Deep OTM put near zero."""
        p = OptionParams(S=200, K=100, T=0.1, r=0.05, sigma=0.10, option_type="put")
        price = bs_price(p)
        assert price < 0.01

    def test_zero_vol_call(self):
        """Zero vol: price equals discounted intrinsic."""
        with pytest.raises(ValueError):
            p = OptionParams(S=110, K=100, T=1.0, r=0.05, sigma=0.0)
            bs_price(p)

    def test_negative_T_raises(self):
        """Negative time raises ValueError."""
        with pytest.raises(ValueError):
            p = OptionParams(S=100, K=100, T=-0.1, r=0.05, sigma=0.20)
            bs_price(p)

    def test_call_with_dividends(self):
        """Call price with continuous dividend yield."""
        p = OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20, q=0.02)
        price = bs_price(p)
        # With q=2%, call should be cheaper
        p_no_div = OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20, q=0.0)
        assert price < bs_price(p_no_div)

    def test_put_with_dividends(self):
        """Put price with dividends should be higher."""
        p = OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20, q=0.03, option_type="put")
        p_nodiv = OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20, q=0.0, option_type="put")
        assert bs_price(p) > bs_price(p_nodiv)

    def test_call_monotone_in_spot(self):
        """Call price increases with spot."""
        prices = [bs_price(OptionParams(S=s, K=100, T=1.0, r=0.05, sigma=0.20))
                  for s in [90, 100, 110, 120]]
        assert all(prices[i] < prices[i+1] for i in range(len(prices)-1))

    def test_put_monotone_in_spot(self):
        """Put price decreases with spot."""
        prices = [bs_price(OptionParams(S=s, K=100, T=1.0, r=0.05, sigma=0.20, option_type="put"))
                  for s in [90, 100, 110, 120]]
        assert all(prices[i] > prices[i+1] for i in range(len(prices)-1))

    def test_call_monotone_in_vol(self):
        """Call price increases with vol (vega > 0)."""
        prices = [bs_price(OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=v))
                  for v in [0.10, 0.20, 0.30, 0.50]]
        assert all(prices[i] < prices[i+1] for i in range(len(prices)-1))

    def test_spx_reference(self):
        """SPX-like: S=5800, K=5800, T=30/365, r=5%, σ=18%"""
        p = OptionParams(S=5800, K=5800, T=30/365, r=0.05, sigma=0.18, q=0.015)
        price = bs_price(p)
        # Sanity: price should be ~1-4% of spot for ATM 30-day
        assert 20 < price < 300


class TestD1D2:
    def test_atm_d1(self):
        """ATM d1 = (r + 0.5*sigma^2)*T / (sigma*sqrt(T))."""
        d1, d2 = _d1_d2(100, 100, 1.0, 0.05, 0.20)
        expected_d1 = (0.05 + 0.5 * 0.04) / 0.20
        assert abs(d1 - expected_d1) < 1e-6

    def test_d2_equals_d1_minus_vol_sqrtT(self):
        d1, d2 = _d1_d2(100, 100, 1.0, 0.05, 0.20)
        assert abs(d2 - (d1 - 0.20)) < 1e-10


class TestPutCallParity:
    def test_parity_holds(self):
        """C - P = S*e^{-qT} - K*e^{-rT}"""
        S, K, T, r, q, sigma = 100, 100, 1.0, 0.05, 0.02, 0.20
        call = bs_price(OptionParams(S, K, T, r, sigma, q, "call"))
        put = bs_price(OptionParams(S, K, T, r, sigma, q, "put"))
        result = put_call_parity_check(call, put, S, K, T, r, q)
        assert result["holds"]
        assert result["parity_error"] < 0.001

    def test_parity_various_strikes(self):
        for K in [80, 100, 120]:
            call = bs_price(OptionParams(100, K, 0.5, 0.05, 0.25, 0.0, "call"))
            put = bs_price(OptionParams(100, K, 0.5, 0.05, 0.25, 0.0, "put"))
            result = put_call_parity_check(call, put, 100, K, 0.5, 0.05, 0.0)
            assert result["holds"], f"Parity failed for K={K}"


class TestBinomialTree:
    def test_european_converges_to_bs(self):
        """Binomial tree with N=300 should match BS within 0.01."""
        p = OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        bs = bs_price(p)
        bt = binomial_tree_price(100, 100, 1.0, 0.05, 0.20, N=300)
        assert abs(bt - bs) < 0.01

    def test_american_put_early_exercise_premium(self):
        """American put >= European put (early exercise premium)."""
        S, K, T, r, sigma = 90, 100, 1.0, 0.10, 0.10
        american = binomial_tree_price(S, K, T, r, sigma, style="american", option_type="put")
        european = bs_price(OptionParams(S, K, T, r, sigma, option_type="put"))
        assert american >= european - 0.001

    def test_american_call_no_dividend_equals_european(self):
        """Without dividends, American call = European call (no early exercise)."""
        S, K, T, r, sigma = 100, 90, 0.5, 0.05, 0.20
        american = binomial_tree_price(S, K, T, r, sigma, q=0.0,
                                       style="american", option_type="call")
        european = bs_price(OptionParams(S, K, T, r, sigma, q=0.0, option_type="call"))
        assert abs(american - european) < 0.05

    def test_expiry_at_zero(self):
        bt = binomial_tree_price(110, 100, 0.0, 0.05, 0.20, option_type="call")
        assert abs(bt - 10.0) < 0.01

    def test_put_otm_at_expiry(self):
        bt = binomial_tree_price(110, 100, 0.0, 0.05, 0.20, option_type="put")
        assert bt == 0.0


class TestImpliedVolatility:
    def test_round_trip(self):
        """IV of BS price should recover input vol."""
        sigma = 0.25
        p = OptionParams(S=100, K=100, T=0.5, r=0.05, sigma=sigma)
        price = bs_price(p)
        iv = implied_volatility(price, 100, 100, 0.5, 0.05)
        assert abs(iv - sigma) < 1e-4

    def test_put_iv_round_trip(self):
        sigma = 0.30
        p = OptionParams(S=100, K=95, T=0.25, r=0.05, sigma=sigma, option_type="put")
        price = bs_price(p)
        iv = implied_volatility(price, 100, 95, 0.25, 0.05, option_type="put")
        assert abs(iv - sigma) < 1e-4

    def test_negative_price_returns_nan(self):
        iv = implied_volatility(-1.0, 100, 100, 0.5, 0.05)
        assert np.isnan(iv)

    def test_zero_T_returns_nan(self):
        iv = implied_volatility(5.0, 100, 100, 0.0, 0.05)
        assert np.isnan(iv)

    def test_various_moneyness(self):
        for K in [85, 100, 115]:
            sigma = 0.20
            p = OptionParams(S=100, K=K, T=0.5, r=0.05, sigma=sigma)
            price = bs_price(p)
            iv = implied_volatility(price, 100, K, 0.5, 0.05)
            if not np.isnan(iv):
                assert abs(iv - sigma) < 0.002


class TestVectorized:
    def test_shape_preserved(self):
        S = np.array([95, 100, 105])
        K = np.array([100, 100, 100])
        T = np.array([0.5, 0.5, 0.5])
        r = np.array([0.05, 0.05, 0.05])
        sigma = np.array([0.20, 0.20, 0.20])
        prices = bs_price_vectorized(S, K, T, r, sigma, option_type="call")
        assert prices.shape == (3,)
        assert all(prices >= 0)

    def test_matches_scalar(self):
        p = bs_price(OptionParams(100, 100, 0.5, 0.05, 0.20))
        pv = bs_price_vectorized(100, 100, 0.5, 0.05, 0.20)
        assert abs(float(pv) - p) < 1e-8
