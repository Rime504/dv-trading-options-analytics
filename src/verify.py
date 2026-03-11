import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

from black_scholes import bs_price, OptionParams
from greeks import compute_greeks
from vol_surface import VolSurface
from backtester import OptionsBacktester, straddle
from data import _synthetic_price_series

# BS accuracy
p = OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
price = bs_price(p)
diff = abs(price - 10.4506)
print(f"BS ATM Call: ${price:.4f} (Bloomberg ref: $10.4506, diff={diff:.6f})")
assert diff < 0.01, f"BS accuracy fail: {diff}"

# Greeks
g = compute_greeks(p)
print(f"Delta={g.delta:.4f} Gamma={g.gamma:.6f} Vega=${g.vega:.4f} Theta=${g.theta:.4f}/day")
assert 0.45 < g.delta < 0.65
assert g.gamma > 0
assert g.vega > 0
assert g.theta < 0

# Vol surface speed
surf = VolSurface(S=5800, r=0.05)
t0 = time.time()
surf.build_synthetic(n_strikes=50, n_expiries=20)
elapsed = time.time() - t0
print(f"Vol surface build: {elapsed:.3f}s (limit: 5s)")
assert elapsed < 5.0, f"Too slow: {elapsed:.2f}s"

# Backtest
prices = _synthetic_price_series("2020-01-01", "2026-03-10")
bt = OptionsBacktester(prices, r=0.05)
result = bt.run(straddle(30))
print(f"SPX Straddle 2020-2026: P&L=${result.total_return:,.0f} | Sharpe={result.sharpe:.2f} | WinRate={result.win_rate:.1f}% | Trades={result.num_trades}")
assert result.num_trades > 0

print("\nALL PRODUCTION CHECKS PASSED")
