"""
Tests for data.py - yfinance wrappers, synthetic data, vol estimators.
"""

import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data import (
    fetch_spot_price, fetch_price_history, fetch_options_chain,
    fetch_futures_curve, fetch_vix, fetch_risk_free_rate,
    _synthetic_price_series, _synthetic_chain, _synthetic_vix,
    _synthetic_spot, parkinson_vol, garman_klass_vol, vol_of_vol,
    vrp_series, export_to_excel, build_report_data
)


class TestSyntheticData:
    def test_price_series_length(self):
        prices = _synthetic_price_series("2020-01-01", "2022-12-31")
        assert len(prices) > 500

    def test_price_series_positive(self):
        prices = _synthetic_price_series()
        assert (prices > 0).all()

    def test_price_series_has_datetime_index(self):
        prices = _synthetic_price_series()
        assert hasattr(prices.index, 'freq') or isinstance(prices.index, pd.DatetimeIndex)

    def test_price_series_covid_crash(self):
        """Prices should be below pre-Feb peak at some point during March 2020."""
        prices = _synthetic_price_series("2020-01-01", "2020-12-31")
        jan_peak = float(prices["2020-01":"2020-02"].max())
        mar_min = float(prices["2020-03"].min())
        # March prices should be below Jan/Feb peak (crash occurred)
        assert mar_min < jan_peak

    def test_synthetic_chain_not_empty(self):
        chain = _synthetic_chain()
        assert len(chain) > 0

    def test_synthetic_chain_structure(self):
        chain = _synthetic_chain()
        for item in chain[:3]:
            assert "K" in item
            assert "T" in item
            assert "mid_price" in item
            assert "iv" in item
            assert item["mid_price"] > 0
            assert item["iv"] > 0

    def test_synthetic_chain_both_types(self):
        chain = _synthetic_chain()
        types = {item["option_type"] for item in chain}
        assert "call" in types
        assert "put" in types

    def test_synthetic_vix(self):
        vix = _synthetic_vix("2020-01-01", "2021-12-31")
        assert len(vix) > 0
        assert (vix > 0).all()

    def test_synthetic_spot(self):
        assert _synthetic_spot("SPY") == 580.0
        assert _synthetic_spot("UNKNOWN") == 400.0


class TestFetchFunctions:
    def test_fetch_spot_price_positive(self):
        spot = fetch_spot_price("SPY")
        assert spot > 0

    def test_fetch_price_history_returns_series(self):
        prices = fetch_price_history("SPY", start="2023-01-01")
        assert isinstance(prices, pd.Series)
        assert len(prices) > 0

    def test_fetch_options_chain_list(self):
        chain = fetch_options_chain("SPY", max_expiries=2)
        assert isinstance(chain, list)
        assert len(chain) > 0

    def test_fetch_futures_curve(self):
        df = fetch_futures_curve("ES", 6)
        assert isinstance(df, pd.DataFrame)
        assert "price" in df.columns
        assert len(df) == 6

    def test_futures_curve_prices_positive(self):
        df = fetch_futures_curve("ES", 4)
        assert (df["price"] > 0).all()

    def test_futures_contango_detected(self):
        df = fetch_futures_curve("ES", 4)
        assert "structure" in df.columns

    def test_fetch_vix_returns_series(self):
        vix = fetch_vix("2020-01-01", "2021-12-31")
        # Accept both Series and single-column DataFrame
        assert isinstance(vix, (pd.Series, pd.DataFrame))
        assert len(vix) > 0

    def test_fetch_risk_free_rate_range(self):
        r = fetch_risk_free_rate()
        assert 0 <= r <= 0.20  # Reasonable range

    def test_fetch_options_chain_structure(self):
        chain = fetch_options_chain("SPY", max_expiries=2)
        for item in chain[:3]:
            assert "K" in item
            assert "T" in item
            assert "mid_price" in item


class TestVolEstimators:
    @pytest.fixture
    def ohlc(self):
        n = 100
        np.random.seed(42)
        close = 100 * np.cumprod(1 + np.random.randn(n) * 0.01)
        high = close * (1 + np.abs(np.random.randn(n) * 0.005))
        low = close * (1 - np.abs(np.random.randn(n) * 0.005))
        open_ = close * (1 + np.random.randn(n) * 0.003)
        idx = pd.date_range("2023-01-01", periods=n, freq="B")
        return (pd.Series(open_, idx), pd.Series(high, idx),
                pd.Series(low, idx), pd.Series(close, idx))

    def test_parkinson_vol_positive(self, ohlc):
        _, high, low, _ = ohlc
        pv = parkinson_vol(high, low, window=21)
        valid = pv.dropna()
        assert (valid > 0).all()

    def test_garman_klass_vol_positive(self, ohlc):
        open_, high, low, close = ohlc
        gkv = garman_klass_vol(open_, high, low, close, window=21)
        valid = gkv.dropna()
        assert (valid > 0).all()

    def test_vol_of_vol_series(self, ohlc):
        _, _, _, close = ohlc
        log_ret = np.log(close / close.shift(1))
        rv = log_ret.rolling(10).std() * np.sqrt(252)
        vov = vol_of_vol(rv)
        assert isinstance(vov, pd.Series)

    def test_vrp_series(self, ohlc):
        _, _, _, close = ohlc
        log_ret = np.log(close / close.shift(1))
        rv = log_ret.rolling(10).std() * np.sqrt(252)
        iv = rv * 1.1 + 0.02
        vrp = vrp_series(iv, rv)
        assert isinstance(vrp, pd.Series)


class TestExport:
    def test_build_report_data(self):
        data = build_report_data("SPY", start="2023-01-01")
        assert "Price_History" in data
        assert isinstance(data["Price_History"], pd.DataFrame)

    def test_export_excel(self, tmp_path):
        data = {"Sheet1": pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})}
        path = str(tmp_path / "test.xlsx")
        export_to_excel(data, path)
        assert os.path.exists(path)
