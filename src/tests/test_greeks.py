"""
Tests for greeks.py - Delta, Gamma, Vega, Theta, Rho, Vanna, Volga, Charm, Speed
Numerical differentiation cross-checks for accuracy.
"""

import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from black_scholes import bs_price, OptionParams
from greeks import compute_greeks, greeks_grid, portfolio_greeks, dollar_greeks


def numerical_delta(p: OptionParams, dS: float = 0.01) -> float:
    p_up = OptionParams(p.S + dS, p.K, p.T, p.r, p.sigma, p.q, p.option_type)
    p_dn = OptionParams(p.S - dS, p.K, p.T, p.r, p.sigma, p.q, p.option_type)
    return (bs_price(p_up) - bs_price(p_dn)) / (2 * dS)


def numerical_gamma(p: OptionParams, dS: float = 0.01) -> float:
    p_up = OptionParams(p.S + dS, p.K, p.T, p.r, p.sigma, p.q, p.option_type)
    p_mid = p
    p_dn = OptionParams(p.S - dS, p.K, p.T, p.r, p.sigma, p.q, p.option_type)
    return (bs_price(p_up) - 2 * bs_price(p_mid) + bs_price(p_dn)) / (dS ** 2)


def numerical_vega(p: OptionParams, dsig: float = 0.0001) -> float:
    p_up = OptionParams(p.S, p.K, p.T, p.r, p.sigma + dsig, p.q, p.option_type)
    p_dn = OptionParams(p.S, p.K, p.T, p.r, p.sigma - dsig, p.q, p.option_type)
    return (bs_price(p_up) - bs_price(p_dn)) / (2 * dsig) / 100.0


def numerical_theta(p: OptionParams, dT: float = 1/365) -> float:
    p_dn = OptionParams(p.S, p.K, max(p.T - dT, 1e-6), p.r, p.sigma, p.q, p.option_type)
    return (bs_price(p_dn) - bs_price(p)) / 1.0  # per calendar day (dT=1/365)


class TestDelta:
    def test_call_delta_atm(self):
        """ATM call delta should be near 0.5."""
        p = OptionParams(S=100, K=100, T=1.0, r=0.0, sigma=0.20)
        g = compute_greeks(p)
        assert 0.45 < g.delta < 0.60

    def test_put_delta_atm(self):
        """ATM put delta should be near -0.5."""
        p = OptionParams(S=100, K=100, T=1.0, r=0.0, sigma=0.20, option_type="put")
        g = compute_greeks(p)
        assert -0.60 < g.delta < -0.40

    def test_delta_put_call_symmetry(self):
        """Delta_call - Delta_put = e^{-qT} (for q=0, = 1)."""
        p_call = OptionParams(S=100, K=100, T=0.5, r=0.0, sigma=0.25, q=0.0)
        p_put = OptionParams(S=100, K=100, T=0.5, r=0.0, sigma=0.25, q=0.0, option_type="put")
        dc = compute_greeks(p_call).delta
        dp = compute_greeks(p_put).delta
        # dc = N(d1), dp = N(d1) - 1 => dc - dp = 1 (when q=0)
        assert abs(dc - dp - 1.0) < 0.001

    def test_deep_itm_call_delta_near_one(self):
        p = OptionParams(S=200, K=100, T=0.1, r=0.05, sigma=0.20)
        g = compute_greeks(p)
        assert g.delta > 0.95

    def test_deep_otm_call_delta_near_zero(self):
        p = OptionParams(S=100, K=200, T=0.1, r=0.05, sigma=0.20)
        g = compute_greeks(p)
        assert g.delta < 0.05

    def test_delta_numerical_match(self):
        """Analytic delta matches numerical finite difference."""
        p = OptionParams(S=100, K=105, T=0.5, r=0.05, sigma=0.25)
        analytic = compute_greeks(p).delta
        numerical = numerical_delta(p)
        assert abs(analytic - numerical) < 0.001

    def test_put_delta_numerical_match(self):
        p = OptionParams(S=100, K=95, T=0.5, r=0.05, sigma=0.25, option_type="put")
        analytic = compute_greeks(p).delta
        numerical = numerical_delta(p)
        assert abs(analytic - numerical) < 0.001


class TestGamma:
    def test_gamma_positive(self):
        """Gamma is always positive for long options."""
        for opt_type in ["call", "put"]:
            p = OptionParams(S=100, K=100, T=0.5, r=0.05, sigma=0.20, option_type=opt_type)
            g = compute_greeks(p)
            assert g.gamma > 0

    def test_gamma_atm_max(self):
        """Gamma is maximized near ATM."""
        g_atm = compute_greeks(OptionParams(100, 100, 0.5, 0.05, 0.20)).gamma
        g_itm = compute_greeks(OptionParams(100, 80, 0.5, 0.05, 0.20)).gamma
        g_otm = compute_greeks(OptionParams(100, 120, 0.5, 0.05, 0.20)).gamma
        assert g_atm > g_itm
        assert g_atm > g_otm

    def test_gamma_numerical(self):
        p = OptionParams(S=100, K=100, T=0.5, r=0.05, sigma=0.20)
        analytic = compute_greeks(p).gamma
        numerical = numerical_gamma(p)
        assert abs(analytic - numerical) < 0.001

    def test_gamma_call_put_equal(self):
        """Gamma is the same for calls and puts (same params)."""
        pc = OptionParams(S=100, K=100, T=0.5, r=0.05, sigma=0.20)
        pp = OptionParams(S=100, K=100, T=0.5, r=0.05, sigma=0.20, option_type="put")
        assert abs(compute_greeks(pc).gamma - compute_greeks(pp).gamma) < 1e-8


class TestVega:
    def test_vega_positive(self):
        for opt in ["call", "put"]:
            p = OptionParams(100, 100, 0.5, 0.05, 0.20, option_type=opt)
            assert compute_greeks(p).vega > 0

    def test_vega_numerical(self):
        p = OptionParams(100, 100, 0.5, 0.05, 0.20)
        analytic = compute_greeks(p).vega
        numerical = numerical_vega(p)
        assert abs(analytic - numerical) < 0.01

    def test_vega_call_put_equal(self):
        pc = OptionParams(100, 100, 0.5, 0.05, 0.20)
        pp = OptionParams(100, 100, 0.5, 0.05, 0.20, option_type="put")
        assert abs(compute_greeks(pc).vega - compute_greeks(pp).vega) < 1e-6

    def test_vega_increases_with_T(self):
        vegas = [compute_greeks(OptionParams(100, 100, T, 0.05, 0.20)).vega
                 for T in [0.1, 0.25, 0.5, 1.0]]
        assert all(vegas[i] < vegas[i+1] for i in range(len(vegas)-1))


class TestTheta:
    def test_theta_negative(self):
        """Long option theta is negative (time decay)."""
        for opt in ["call", "put"]:
            p = OptionParams(100, 100, 0.5, 0.05, 0.20, option_type=opt)
            assert compute_greeks(p).theta < 0

    def test_theta_atm_largest_decay(self):
        """ATM options have largest theta decay."""
        th_atm = abs(compute_greeks(OptionParams(100, 100, 0.25, 0.05, 0.20)).theta)
        th_otm = abs(compute_greeks(OptionParams(100, 130, 0.25, 0.05, 0.20)).theta)
        assert th_atm > th_otm


class TestHigherGreeks:
    def test_vanna_sign(self):
        """Vanna (dDelta/dVol) should be positive for OTM calls."""
        p = OptionParams(100, 110, 0.5, 0.05, 0.20)  # OTM call
        g = compute_greeks(p)
        assert g.vanna > 0

    def test_volga_positive(self):
        """Volga (dVega/dVol) is always positive."""
        for K in [85, 100, 115]:
            p = OptionParams(100, K, 0.5, 0.05, 0.20)
            assert compute_greeks(p).volga >= 0

    def test_at_expiry_greeks(self):
        """At expiry, most greeks collapse."""
        p = OptionParams(100, 100, 1e-12, 0.05, 0.20)
        g = compute_greeks(p)
        assert g.gamma == 0
        assert g.vega == 0


class TestPortfolioGreeks:
    def test_straddle_delta_near_zero(self):
        """ATM straddle net delta = e^{-qT} (call delta + put delta, q=r=0 gives ~0)."""
        p_call = OptionParams(100, 100, 0.5, 0.0, 0.20, 0.0, "call")
        p_put = OptionParams(100, 100, 0.5, 0.0, 0.20, 0.0, "put")
        agg = portfolio_greeks([(p_call, 1), (p_put, 1)])
        # With r=q=0: dc + dp = N(d1) + (N(d1)-1) = 2*N(d1) - 1, near 0 for ATM
        assert abs(agg["delta"]) < 0.15

    def test_short_straddle_negative_gamma(self):
        """Short straddle has negative gamma."""
        p_call = OptionParams(100, 100, 0.5, 0.05, 0.20, 0.0, "call")
        p_put = OptionParams(100, 100, 0.5, 0.05, 0.20, 0.0, "put")
        agg = portfolio_greeks([(p_call, -1), (p_put, -1)])
        assert agg["gamma"] < 0

    def test_additive_greeks(self):
        """Portfolio greeks are sum of individual greeks."""
        p1 = OptionParams(100, 100, 0.5, 0.05, 0.20, 0.0, "call")
        p2 = OptionParams(100, 105, 0.5, 0.05, 0.20, 0.0, "put")
        agg = portfolio_greeks([(p1, 2), (p2, -1)])
        g1 = compute_greeks(p1)
        g2 = compute_greeks(p2)
        expected_delta = g1.delta * 2 + g2.delta * (-1)
        assert abs(agg["delta"] - expected_delta) < 1e-8


class TestDollarGreeks:
    def test_dollar_delta_positive_long_call(self):
        p = OptionParams(100, 100, 0.5, 0.05, 0.20, 0.0, "call")
        dg = dollar_greeks(p, notional=1.0)
        assert dg["delta_dollars"] > 0

    def test_dollar_vega_returned(self):
        p = OptionParams(100, 100, 0.5, 0.05, 0.20, 0.0, "call")
        dg = dollar_greeks(p, notional=100.0)
        assert "vega_dollars" in dg
        assert dg["vega_dollars"] > 0


class TestGreeksGrid:
    def test_grid_shape(self):
        K_range = np.linspace(90, 110, 5)
        T_range = np.linspace(0.1, 1.0, 5)
        grid = greeks_grid(100, K_range, T_range, 0.05, 0.20)
        assert grid["delta"].shape == (5, 5)
        assert grid["gamma"].shape == (5, 5)

    def test_grid_delta_monotone_in_strike(self):
        """Delta decreases as strike increases (call)."""
        K_range = np.array([90, 95, 100, 105, 110])
        T_range = np.array([0.5])
        grid = greeks_grid(100, K_range, T_range, 0.05, 0.20)
        deltas = grid["delta"][:, 0]
        assert all(deltas[i] > deltas[i+1] for i in range(len(deltas)-1))
