"""
DV Trading Options Analytics Suite - Streamlit Dashboard
Deploy: streamlit run dashboard.py
"""

import sys
import os
# Ensure src/ is on path whether running locally or via Streamlit Cloud
_src = os.path.dirname(os.path.abspath(__file__))
if _src not in sys.path:
    sys.path.insert(0, _src)

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from black_scholes import bs_price, binomial_tree_price, implied_volatility, OptionParams, put_call_parity_check
from greeks import compute_greeks, greeks_grid
from vol_surface import VolSurface, svi_implied_vol, fit_svi_slice, term_structure
from backtester import (
    OptionsBacktester, straddle, strangle, short_straddle,
    iron_condor, bull_call_spread, monte_carlo_var
)
from data import (
    fetch_spot_price, fetch_price_history, fetch_options_chain,
    fetch_futures_curve, fetch_vix, fetch_risk_free_rate,
    _synthetic_chain, _synthetic_price_series, export_to_excel
)

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="DV Trading Options Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for mobile-responsive dark theme
st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; max-width: 100%; }
    .metric-card {
        background: #1e1e2e; border-radius: 8px; padding: 1rem;
        border-left: 4px solid #7c3aed; margin-bottom: 0.5rem;
    }
    .stMetric label { font-size: 0.8rem !important; color: #94a3b8 !important; }
    .stMetric value { font-size: 1.4rem !important; }
    @media (max-width: 768px) {
        .stColumn { width: 100% !important; }
    }
    h1, h2, h3 { color: #e2e8f0; }
    .stSelectbox label, .stSlider label { color: #94a3b8; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## DV Trading\n### Options Analytics Suite")
    st.markdown("---")

    page = st.selectbox(
        "Navigation",
        [
            "Black-Scholes Pricer",
            "Greeks Calculator",
            "Volatility Surface",
            "Options Backtester",
            "Futures Curve",
            "Risk Dashboard",
        ],
    )

    st.markdown("---")
    st.markdown("**Market Data**")
    ticker = st.selectbox("Ticker", ["SPY", "SPX", "QQQ", "IWM", "AAPL", "TSLA"])

    @st.cache_data(ttl=300)
    def get_spot(t):
        return fetch_spot_price(t)

    @st.cache_data(ttl=300)
    def get_rf():
        return fetch_risk_free_rate()

    spot = get_spot(ticker)
    r_default = get_rf()

    st.metric("Spot", f"${spot:.2f}")
    st.metric("Risk-Free Rate", f"{r_default*100:.2f}%")
    st.markdown("---")
    st.caption("Data: yfinance (free) | Model: Black-Scholes / SVI")
    st.caption("v1.0 | DV Trading Intern Project")


# ===========================================================================
# PAGE 1: Black-Scholes Pricer
# ===========================================================================

if page == "Black-Scholes Pricer":
    st.title("Black-Scholes Option Pricer")
    st.markdown("Analytic European + CRR Binomial Tree (American/European)")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        S = st.number_input("Spot Price (S)", value=float(spot), min_value=0.01, step=1.0)
        K = st.number_input("Strike Price (K)", value=float(round(spot, 0)), min_value=0.01, step=1.0)
        T_days = st.slider("Days to Expiry", 1, 730, 30)
        T = T_days / 365.0

    with col2:
        r = st.number_input("Risk-Free Rate (%)", value=float(r_default * 100), step=0.1, min_value=0.0) / 100
        sigma = st.slider("Implied Volatility (%)", 1, 200, 20) / 100
        q = st.number_input("Dividend Yield (%)", value=1.5, step=0.1, min_value=0.0) / 100

    with col3:
        opt_type = st.selectbox("Option Type", ["call", "put"])
        style = st.selectbox("Exercise Style", ["european", "american"])
        n_steps = st.slider("Binomial Steps", 50, 500, 200)

    p = OptionParams(S=S, K=K, T=T, r=r, sigma=sigma, q=q, option_type=opt_type)
    bs = bs_price(p)
    bt = binomial_tree_price(S, K, T, r, sigma, q, opt_type, style, n_steps)

    moneyness = S / K
    intrinsic = max(S - K, 0) if opt_type == "call" else max(K - S, 0)
    time_value = bs - intrinsic

    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("BS Price", f"${bs:.4f}")
    c2.metric("Binomial Tree", f"${bt:.4f}", f"Diff: {(bt-bs):.4f}")
    c3.metric("Moneyness (S/K)", f"{moneyness:.4f}")
    c4.metric("Intrinsic Value", f"${intrinsic:.4f}")
    c5.metric("Time Value", f"${time_value:.4f}")

    # Put-Call Parity
    p_put = OptionParams(S=S, K=K, T=T, r=r, sigma=sigma, q=q, option_type="put")
    p_call = OptionParams(S=S, K=K, T=T, r=r, sigma=sigma, q=q, option_type="call")
    parity = put_call_parity_check(bs_price(p_call), bs_price(p_put), S, K, T, r, q)
    st.info(f"Put-Call Parity: C-P={parity['lhs']:.4f} | Se^{{-qT}}-Ke^{{-rT}}={parity['rhs']:.4f} | Error={parity['parity_error']:.6f} | {'✓ Holds' if parity['holds'] else '✗ Violated'}")

    st.markdown("---")
    # Price vs. Strike heatmap
    st.subheader("Price vs Strike & Volatility")
    tab1, tab2, tab3 = st.tabs(["Price vs Strike", "Price vs Vol", "P&L Payoff"])

    with tab1:
        strikes = np.linspace(S * 0.7, S * 1.3, 100)
        prices = [bs_price(OptionParams(S, k, T, r, sigma, q, opt_type)) for k in strikes]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=strikes, y=prices, name="BS Price", line=dict(color="#7c3aed", width=2)))
        intrinsics = [max(S - k, 0) if opt_type == "call" else max(k - S, 0) for k in strikes]
        fig.add_trace(go.Scatter(x=strikes, y=intrinsics, name="Intrinsic", line=dict(color="#f59e0b", dash="dash")))
        fig.add_vline(x=K, line_dash="dot", annotation_text=f"K={K}", line_color="#e74c3c")
        fig.update_layout(title="Option Price vs Strike", xaxis_title="Strike", yaxis_title="Price ($)",
                          template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        vols = np.linspace(0.05, 1.0, 100)
        prices_vs_vol = [bs_price(OptionParams(S, K, T, r, v, q, opt_type)) for v in vols]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=vols * 100, y=prices_vs_vol, line=dict(color="#10b981", width=2)))
        fig2.add_vline(x=sigma * 100, line_dash="dot", annotation_text=f"σ={sigma*100:.0f}%")
        fig2.update_layout(title="Option Price vs Volatility", xaxis_title="Implied Vol (%)",
                           yaxis_title="Price ($)", template="plotly_dark", height=400)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        spots_grid = np.linspace(S * 0.7, S * 1.3, 200)
        payoff = [max(s - K, 0) if opt_type == "call" else max(K - s, 0) for s in spots_grid]
        pnl = [p_ - bs for p_ in payoff]
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=spots_grid, y=pnl, fill="tozeroy",
                                  line=dict(color="#3b82f6"), name="P&L at Expiry"))
        fig3.add_hline(y=0, line_color="white", line_dash="dash")
        fig3.add_vline(x=S, annotation_text="Current Spot", line_color="#f59e0b")
        fig3.update_layout(title="P&L at Expiry", xaxis_title="Underlying Price",
                           yaxis_title="P&L ($)", template="plotly_dark", height=400)
        st.plotly_chart(fig3, use_container_width=True)


# ===========================================================================
# PAGE 2: Greeks Calculator
# ===========================================================================

elif page == "Greeks Calculator":
    st.title("Options Greeks Calculator")
    st.markdown("Delta | Gamma | Vega | Theta | Rho | Vanna | Volga | Charm | Speed")

    col1, col2 = st.columns([1, 2])

    with col1:
        S_g = st.number_input("Spot (S)", value=float(spot), step=1.0)
        K_g = st.number_input("Strike (K)", value=float(round(spot, 0)), step=1.0)
        T_g = st.slider("DTE", 1, 730, 30) / 365.0
        r_g = st.number_input("Rate (%)", value=float(r_default * 100), step=0.1) / 100
        sig_g = st.slider("IV (%)", 1, 150, 20) / 100
        q_g = st.number_input("Div Yield (%)", value=1.5, step=0.1) / 100
        opt_g = st.selectbox("Type", ["call", "put"])

    p_g = OptionParams(S=S_g, K=K_g, T=T_g, r=r_g, sigma=sig_g, q=q_g, option_type=opt_g)
    g = compute_greeks(p_g)

    with col2:
        st.subheader("First-Order Greeks")
        gc1, gc2, gc3, gc4 = st.columns(4)
        gc1.metric("Delta", f"{g.delta:.4f}", help="Change in option price per $1 spot move")
        gc2.metric("Gamma", f"{g.gamma:.6f}", help="Change in delta per $1 spot move")
        gc3.metric("Vega", f"${g.vega:.4f}", help="Price change per 1% vol increase")
        gc4.metric("Theta", f"${g.theta:.4f}/day", help="Price decay per calendar day")

        st.subheader("Higher-Order Greeks")
        gc5, gc6, gc7, gc8, gc9 = st.columns(5)
        gc5.metric("Rho", f"${g.rho:.4f}", help="Sensitivity to 1% rate change")
        gc6.metric("Vanna", f"{g.vanna:.6f}", help="dDelta/dVol")
        gc7.metric("Volga", f"${g.volga:.6f}", help="dVega/dVol (Vomma)")
        gc8.metric("Charm", f"{g.charm:.6f}/day", help="Delta decay rate")
        gc9.metric("Speed", f"{g.speed:.8f}", help="dGamma/dS")

    st.markdown("---")
    st.subheader("Greeks Heatmap (Strike x Expiry)")

    K_range = np.linspace(S_g * 0.8, S_g * 1.2, 20)
    T_range = np.linspace(0.05, 1.0, 20)
    greek_name = st.selectbox("Greek to Plot", ["delta", "gamma", "vega", "theta", "vanna", "volga"])

    grid = greeks_grid(S_g, K_range, T_range, r_g, sig_g, q_g, opt_g)
    values = grid[greek_name]

    fig_hm = go.Figure(data=go.Heatmap(
        z=values,
        x=[f"{t*365:.0f}d" for t in T_range],
        y=[f"{k:.0f}" for k in K_range],
        colorscale="RdBu_r" if greek_name in ["delta", "theta"] else "Viridis",
        text=np.round(values, 4),
        texttemplate="%{text}",
        textfont={"size": 9},
        colorbar=dict(title=greek_name.capitalize()),
    ))
    fig_hm.update_layout(
        title=f"{greek_name.capitalize()} Heatmap: Strike vs Expiry",
        xaxis_title="Days to Expiry", yaxis_title="Strike",
        template="plotly_dark", height=500,
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    # Greeks P&L Attribution
    st.subheader("Taylor Series P&L Attribution")
    dS = st.slider("Spot Change ($)", -50, 50, 0)
    dV = st.slider("Vol Change (%)", -10, 10, 0) / 100
    dT = st.slider("Days Passed", 0, 30, 1)

    pnl_delta = g.delta * dS
    pnl_gamma = 0.5 * g.gamma * dS ** 2
    pnl_vega = g.vega * (dV * 100)
    pnl_theta = g.theta * dT
    pnl_total = pnl_delta + pnl_gamma + pnl_vega + pnl_theta

    attr_data = pd.DataFrame({
        "Component": ["Delta", "Gamma", "Vega", "Theta", "Total"],
        "P&L ($)": [pnl_delta, pnl_gamma, pnl_vega, pnl_theta, pnl_total],
    })
    fig_attr = px.bar(attr_data[:-1], x="Component", y="P&L ($)",
                      color="P&L ($)", color_continuous_scale="RdYlGn",
                      template="plotly_dark")
    fig_attr.update_layout(title=f"Greeks P&L Attribution | Total: ${pnl_total:.2f}", height=350)
    st.plotly_chart(fig_attr, use_container_width=True)


# ===========================================================================
# PAGE 3: Volatility Surface
# ===========================================================================

elif page == "Volatility Surface":
    st.title("Implied Volatility Surface")
    st.markdown("SVI Smile Fitting | 3D Interactive Plot | Term Structure")

    @st.cache_data(ttl=300)
    def get_chain(t):
        return fetch_options_chain(t, max_expiries=6)

    with st.spinner("Building vol surface..."):
        chain = get_chain(ticker)
        S_vs = spot

        surf = VolSurface(S=S_vs, r=r_default, q=0.015)
        surf.build_synthetic(
            K_range=(0.75, 1.25),
            T_range=(0.05, 2.0),
            n_strikes=40,
            n_expiries=15,
            atm_vol=0.18,
            skew=-0.1,
            smile=0.05,
        )

    tab1, tab2, tab3 = st.tabs(["3D Surface", "Smile Slices", "Term Structure"])

    with tab1:
        st.subheader("3D Implied Volatility Surface")
        K_grid = surf.K_grid
        T_grid = surf.T_grid
        Z = surf.surface_grid * 100  # to %

        fig_3d = go.Figure(data=[go.Surface(
            x=T_grid * 365,   # days
            y=K_grid,
            z=Z,
            colorscale="Viridis",
            colorbar=dict(title="IV (%)"),
            contours=dict(z=dict(show=True, usecolormap=True)),
        )])
        fig_3d.update_layout(
            title=f"{ticker} Implied Volatility Surface",
            scene=dict(
                xaxis_title="Days to Expiry",
                yaxis_title="Strike",
                zaxis_title="IV (%)",
                bgcolor="#0e1117",
                xaxis=dict(backgroundcolor="#1e1e2e"),
                yaxis=dict(backgroundcolor="#1e1e2e"),
                zaxis=dict(backgroundcolor="#1e1e2e"),
            ),
            template="plotly_dark",
            height=600,
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    with tab2:
        st.subheader("Vol Smile by Expiry")
        T_idx = st.slider("Expiry Slice (index)", 0, len(T_grid) - 1, len(T_grid) // 3)
        T_sel = T_grid[T_idx]

        fig_smile = go.Figure()
        fig_smile.add_trace(go.Scatter(
            x=K_grid,
            y=surf.surface_grid[:, T_idx] * 100,
            mode="lines+markers",
            line=dict(color="#7c3aed", width=2),
            name=f"T={T_sel*365:.0f}d",
        ))
        fig_smile.add_vline(x=S_vs, line_dash="dash", annotation_text="ATM",
                            line_color="#f59e0b")
        fig_smile.update_layout(
            title=f"Vol Smile at {T_sel*365:.0f} Days to Expiry",
            xaxis_title="Strike", yaxis_title="Implied Vol (%)",
            template="plotly_dark", height=400,
        )
        st.plotly_chart(fig_smile, use_container_width=True)

        # SVI fit display
        atm_idx = np.argmin(np.abs(K_grid - S_vs))
        iv_slice = surf.surface_grid[:, T_idx]
        F = S_vs * np.exp((r_default - 0.015) * T_sel)
        k = np.log(K_grid / F)

        try:
            svi_params, rmse = fit_svi_slice(K_grid, iv_slice, F, T_sel)
            iv_svi = svi_implied_vol(k, T_sel, *svi_params) * 100

            fig_smile.add_trace(go.Scatter(
                x=K_grid, y=iv_svi, mode="lines",
                line=dict(color="#10b981", dash="dash", width=2),
                name="SVI Fit",
            ))
            st.plotly_chart(fig_smile, use_container_width=True)
            st.metric("SVI RMSE", f"{rmse*100:.4f}%")
        except Exception:
            pass

    with tab3:
        st.subheader("ATM Vol Term Structure")
        atm_vols = []
        for j, T in enumerate(T_grid):
            atm_idx = np.argmin(np.abs(K_grid - S_vs))
            atm_vols.append(surf.surface_grid[atm_idx, j])

        ts = term_structure(T_grid, np.array(atm_vols))

        fig_ts = make_subplots(rows=2, cols=1, subplot_titles=["ATM Implied Vol", "Forward Vol"])
        fig_ts.add_trace(go.Scatter(
            x=T_grid * 365, y=np.array(atm_vols) * 100,
            mode="lines+markers", line=dict(color="#7c3aed"),
        ), row=1, col=1)
        fig_ts.add_trace(go.Bar(
            x=T_grid[1:] * 365, y=ts["forward_vols"] * 100,
            marker_color="#10b981", name="Forward Vol",
        ), row=2, col=1)
        fig_ts.update_layout(template="plotly_dark", height=500,
                             showlegend=False)
        fig_ts.update_xaxes(title_text="Days to Expiry")
        fig_ts.update_yaxes(title_text="Vol (%)")
        st.plotly_chart(fig_ts, use_container_width=True)

        cal_arb = ts["calendar_arb_free"]
        st.info(f"Calendar Arbitrage Free: {'YES' if cal_arb else 'NO - Check Surface'}")


# ===========================================================================
# PAGE 4: Options Backtester
# ===========================================================================

elif page == "Options Backtester":
    st.title("Options Strategy Backtester")
    st.markdown("Historical P&L | SPX/SPY | COVID Crash Validation | 2020-2026")

    col1, col2 = st.columns([1, 3])

    with col1:
        strategy_name = st.selectbox(
            "Strategy",
            ["ATM Straddle", "Short Straddle", "OTM Strangle", "Iron Condor",
             "Bull Call Spread", "Short Strangle"],
        )
        expiry_dte = st.slider("Entry DTE", 7, 90, 30)
        exit_dte = st.slider("Exit DTE", 0, 20, 5)
        otm_pct = st.slider("OTM Width (%)", 1, 15, 5) / 100
        start_date = st.date_input("Start Date", pd.Timestamp("2020-01-01"))
        end_date = st.date_input("End Date", pd.Timestamp("2026-03-10"))
        vol_scale = st.slider("Vol Scale (VRP adj.)", 0.5, 2.0, 1.0, 0.05)
        run_bt = st.button("Run Backtest", type="primary")

    strategy_map = {
        "ATM Straddle": straddle(expiry_dte),
        "Short Straddle": short_straddle(expiry_dte),
        "OTM Strangle": strangle(otm_pct, expiry_dte),
        "Iron Condor": iron_condor(otm_pct, expiry_dte),
        "Bull Call Spread": bull_call_spread(otm_pct, expiry_dte),
        "Short Strangle": strangle(otm_pct, expiry_dte),
    }

    with col2:
        if run_bt:
            with st.spinner("Running backtest..."):
                @st.cache_data(ttl=3600)
                def get_prices(t, s, e):
                    return fetch_price_history(t, start=s, end=e)

                prices = get_prices(ticker, str(start_date), str(end_date))
                bt_engine = OptionsBacktester(prices, r=r_default)
                strat = strategy_map[strategy_name]
                strat.entry_dte = expiry_dte
                strat.exit_dte = exit_dte

                result = bt_engine.run(strat, vol_scale=vol_scale)

            # Metrics
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("Total P&L", f"${result.total_return:,.0f}")
            mc2.metric("Sharpe", f"{result.sharpe:.2f}")
            mc3.metric("Max Drawdown", f"${result.max_drawdown:,.0f}")
            mc4.metric("Win Rate", f"{result.win_rate:.1f}%")
            mc5.metric("# Trades", result.num_trades)

            # Equity Curve
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                x=result.equity_curve.index,
                y=result.equity_curve.values,
                fill="tozeroy",
                line=dict(color="#7c3aed", width=2),
                name="Cumulative P&L",
            ))
            # COVID region
            fig_eq.add_vrect(x0="2020-02-20", x1="2020-03-23",
                             fillcolor="red", opacity=0.15,
                             annotation_text="COVID Crash")
            fig_eq.update_layout(
                title=f"{strategy_name} Equity Curve | {ticker}",
                xaxis_title="Date", yaxis_title="Cumulative P&L ($)",
                template="plotly_dark", height=400,
            )
            st.plotly_chart(fig_eq, use_container_width=True)

            # Trade distribution
            if not result.trades.empty:
                fig_dist = make_subplots(1, 2,
                    subplot_titles=["P&L Distribution", "Greeks at Entry"])
                fig_dist.add_trace(go.Histogram(
                    x=result.trades["pnl"], nbinsx=30,
                    marker_color="#7c3aed", name="P&L",
                ), 1, 1)
                fig_dist.add_trace(go.Scatter(
                    x=result.trades["entry_date"],
                    y=result.trades["vega_entry"],
                    mode="lines", line=dict(color="#10b981"),
                    name="Vega",
                ), 1, 2)
                fig_dist.update_layout(template="plotly_dark", height=350,
                                       showlegend=True)
                st.plotly_chart(fig_dist, use_container_width=True)

                st.dataframe(
                    result.trades[["entry_date", "exit_date", "S_entry", "S_exit",
                                   "move_pct", "sigma_entry", "pnl", "cumulative_pnl"]].tail(20),
                    use_container_width=True,
                )

                # Export
                buf = result.trades.to_csv(index=False).encode()
                st.download_button("Download Trades CSV", buf, "trades.csv", "text/csv")
        else:
            st.info("Configure parameters and click **Run Backtest** to start.")
            # Show sample COVID backtest
            st.subheader("Quick Preview: SPY ATM Straddle 2020-2021")
            prices_demo = _synthetic_price_series("2020-01-01", "2021-12-31")
            bt_demo = OptionsBacktester(prices_demo)
            res_demo = bt_demo.run(straddle(30))
            fig_demo = go.Figure()
            fig_demo.add_trace(go.Scatter(x=res_demo.equity_curve.index,
                                          y=res_demo.equity_curve.values,
                                          line=dict(color="#10b981")))
            fig_demo.add_vrect(x0="2020-02-20", x1="2020-03-23",
                               fillcolor="red", opacity=0.2,
                               annotation_text="COVID Crash")
            fig_demo.update_layout(title="ATM Straddle Demo (Synthetic Data)",
                                   template="plotly_dark", height=300)
            st.plotly_chart(fig_demo, use_container_width=True)


# ===========================================================================
# PAGE 5: Futures Curve
# ===========================================================================

elif page == "Futures Curve":
    st.title("Futures Curve Analysis")
    st.markdown("Contango / Backwardation | Roll Cost | Basis Analysis")

    @st.cache_data(ttl=300)
    def get_futures():
        return fetch_futures_curve("ES", 8)

    curve_df = get_futures()

    # Structure detection
    is_contango = curve_df["annualised_basis"].iloc[1:].mean() > 0
    structure_label = "CONTANGO" if is_contango else "BACKWARDATION"
    structure_color = "#ef4444" if is_contango else "#10b981"

    st.markdown(f"**Market Structure: <span style='color:{structure_color}'>{structure_label}</span>**",
                unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(
            x=curve_df["days_to_exp"],
            y=curve_df["price"],
            mode="lines+markers+text",
            text=curve_df["contract"],
            textposition="top center",
            line=dict(color="#7c3aed", width=2),
            marker=dict(size=10),
            name="Futures Price",
        ))
        # Spot reference
        es_spot = spot * 10
        fig_curve.add_hline(y=es_spot, line_dash="dash",
                            annotation_text=f"Spot (~${es_spot:,.0f})",
                            line_color="#f59e0b")
        fig_curve.update_layout(
            title="ES Futures Forward Curve",
            xaxis_title="Days to Expiry", yaxis_title="Price",
            template="plotly_dark", height=400,
        )
        st.plotly_chart(fig_curve, use_container_width=True)

        # Basis/Roll
        fig_basis = go.Figure()
        fig_basis.add_trace(go.Bar(
            x=curve_df["contract"],
            y=curve_df["annualised_basis"],
            marker_color=[structure_color] * len(curve_df),
            name="Annualised Basis (%)",
        ))
        fig_basis.update_layout(
            title="Roll/Carry (Annualised Basis %)",
            xaxis_title="Contract", yaxis_title="Basis (% p.a.)",
            template="plotly_dark", height=300,
        )
        st.plotly_chart(fig_basis, use_container_width=True)

    with col2:
        st.subheader("Futures Chain")
        display_df = curve_df[["contract", "price", "expiry", "days_to_exp",
                                "annualised_basis", "structure"]].copy()
        display_df["price"] = display_df["price"].map("${:,.2f}".format)
        display_df["annualised_basis"] = display_df["annualised_basis"].map("{:.2f}%".format)
        st.dataframe(display_df, use_container_width=True)

        st.subheader("Roll Cost Calculator")
        roll_qty = st.number_input("Contracts", 1, 100, 1)
        front_price = float(curve_df["price"].iloc[0])
        back_price = float(curve_df["price"].iloc[1])
        roll_cost = (back_price - front_price) * roll_qty * 50  # $50/point ES
        st.metric("Roll Cost (1 → 2)", f"${roll_cost:,.0f}")

    # VIX term structure
    st.subheader("VIX / Volatility Term Structure")

    @st.cache_data(ttl=300)
    def get_vix_data():
        return fetch_vix("2020-01-01")

    vix_data = get_vix_data()
    fig_vix = go.Figure()
    fig_vix.add_trace(go.Scatter(
        x=vix_data.index, y=vix_data.values,
        fill="tozeroy", line=dict(color="#f59e0b"),
        name="VIX",
    ))
    fig_vix.add_hrect(y0=20, y1=float(vix_data.max()), fillcolor="red", opacity=0.08)
    fig_vix.add_hline(y=20, line_dash="dash", annotation_text="20 (Fear Threshold)")
    fig_vix.add_vrect(x0="2020-02-20", x1="2020-03-18",
                      fillcolor="red", opacity=0.2, annotation_text="COVID VIX Spike")
    fig_vix.update_layout(title="VIX Index (2020-2026)",
                          template="plotly_dark", height=350)
    st.plotly_chart(fig_vix, use_container_width=True)


# ===========================================================================
# PAGE 6: Risk Dashboard
# ===========================================================================

elif page == "Risk Dashboard":
    st.title("Risk Dashboard")
    st.markdown("VaR | Monte Carlo | Portfolio Greeks | Position Sizing")

    st.subheader("Portfolio Builder")

    # Default portfolio
    default_positions = [
        {"type": "call", "K_offset": 0.0, "T_days": 30, "qty": 1},
        {"type": "put", "K_offset": 0.0, "T_days": 30, "qty": 1},
        {"type": "call", "K_offset": 0.05, "T_days": 30, "qty": -2},
    ]

    n_pos = st.slider("Number of Positions", 1, 6, 3)
    positions_params = []

    with st.expander("Configure Positions", expanded=True):
        cols_header = st.columns([2, 2, 2, 2, 1])
        for c, h in zip(cols_header, ["Type", "Strike %", "DTE", "Qty", ""]):
            c.markdown(f"**{h}**")

        for i in range(n_pos):
            default = default_positions[i] if i < len(default_positions) else default_positions[-1]
            pc1, pc2, pc3, pc4, _ = st.columns([2, 2, 2, 2, 1])
            opt_type_i = pc1.selectbox(f"T{i+1}", ["call", "put"], key=f"type_{i}",
                                        index=0 if default["type"] == "call" else 1)
            k_off_i = pc2.number_input(f"K% {i+1}", -20.0, 20.0, float(default["K_offset"] * 100),
                                        step=0.5, key=f"k_{i}") / 100
            dte_i = pc3.number_input(f"DTE {i+1}", 1, 365, default["T_days"], key=f"t_{i}")
            qty_i = pc4.number_input(f"Qty {i+1}", -10, 10, default["qty"], key=f"q_{i}")

            S_r = spot
            K_abs = S_r * (1 + k_off_i)
            T_abs = dte_i / 365.0
            positions_params.append(
                (OptionParams(S=S_r, K=K_abs, T=T_abs, r=r_default,
                               sigma=0.18, q=0.015, option_type=opt_type_i), qty_i)
            )

    # Portfolio Greeks
    st.subheader("Portfolio Greeks")
    agg_greeks = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
    total_value = 0.0
    for p, qty in positions_params:
        if p.T > 0:
            g = compute_greeks(p)
            for key in agg_greeks:
                agg_greeks[key] += getattr(g, key) * qty * 100
            total_value += bs_price(p) * qty * 100

    pg1, pg2, pg3, pg4, pg5 = st.columns(5)
    pg1.metric("Portfolio Value", f"${total_value:,.2f}")
    pg2.metric("Net Delta", f"{agg_greeks['delta']:.2f}")
    pg3.metric("Net Gamma", f"{agg_greeks['gamma']:.4f}")
    pg4.metric("Net Vega", f"${agg_greeks['vega']:,.2f}")
    pg5.metric("Net Theta", f"${agg_greeks['theta']:.2f}/day")

    # VaR
    st.subheader("Value at Risk (Monte Carlo)")
    col_var1, col_var2 = st.columns([1, 2])
    with col_var1:
        n_sims = st.selectbox("Simulations", [1000, 5000, 10000, 50000], index=1)
        conf_level = st.selectbox("Confidence Level", [0.95, 0.99, 0.999], index=1)
        horizon = st.slider("Horizon (days)", 1, 10, 1)
        run_var = st.button("Compute VaR", type="primary")

    with col_var2:
        if run_var:
            with st.spinner("Running Monte Carlo..."):
                var_result = monte_carlo_var(
                    total_value, positions_params,
                    horizon_days=horizon,
                    n_sims=n_sims,
                    confidence=conf_level,
                )

            vc1, vc2, vc3 = st.columns(3)
            vc1.metric(f"VaR ({conf_level*100:.0f}%, {horizon}d)",
                       f"${var_result['var_1day']:,.2f}")
            vc2.metric("CVaR (Expected Shortfall)",
                       f"${var_result['cvar_1day']:,.2f}")
            vc3.metric("Worst Case",
                       f"${var_result['worst_case']:,.2f}")

            # P&L Distribution
            pnls = var_result["pnl_distribution"]
            fig_var = go.Figure()
            fig_var.add_trace(go.Histogram(
                x=pnls, nbinsx=100,
                marker_color="#7c3aed",
                name="Simulated P&L",
            ))
            fig_var.add_vline(x=var_result["var_1day"],
                              line_dash="dash", line_color="#ef4444",
                              annotation_text=f"VaR: ${var_result['var_1day']:,.0f}")
            fig_var.add_vline(x=var_result["cvar_1day"],
                              line_dash="dot", line_color="#f59e0b",
                              annotation_text=f"CVaR: ${var_result['cvar_1day']:,.0f}")
            fig_var.update_layout(
                title=f"Monte Carlo P&L Distribution ({n_sims:,} sims)",
                xaxis_title="P&L ($)", yaxis_title="Frequency",
                template="plotly_dark", height=400,
            )
            st.plotly_chart(fig_var, use_container_width=True)
        else:
            st.info("Click **Compute VaR** to run Monte Carlo simulation.")

    # Stress Tests
    st.subheader("Stress Test Scenarios")
    scenarios = {
        "COVID Crash (-35%)": -0.35,
        "2022 Rate Hike (-25%)": -0.25,
        "Flash Crash (-10%)": -0.10,
        "Bull Run (+20%)": +0.20,
        "+1 StdDev": +0.18,
        "-1 StdDev": -0.18,
    }

    stress_results = []
    for scenario, move in scenarios.items():
        s_pnl = 0.0
        for p_orig, qty in positions_params:
            S_new = p_orig.S * (1 + move)
            sigma_new = p_orig.sigma * (1 + abs(move) * 2) if move < 0 else p_orig.sigma * 0.8
            p_new = OptionParams(S=S_new, K=p_orig.K, T=p_orig.T,
                                 r=p_orig.r, sigma=max(sigma_new, 0.01),
                                 q=p_orig.q, option_type=p_orig.option_type)
            if p_orig.T > 0:
                s_pnl += (bs_price(p_new) - bs_price(p_orig)) * qty * 100
        stress_results.append({"Scenario": scenario, "Move": f"{move*100:+.0f}%", "P&L": f"${s_pnl:,.2f}", "PnL_raw": s_pnl})

    stress_df = pd.DataFrame(stress_results)
    fig_stress = px.bar(
        stress_df, x="Scenario", y="PnL_raw",
        color="PnL_raw", color_continuous_scale="RdYlGn",
        template="plotly_dark",
    )
    fig_stress.update_layout(title="Stress Test P&L", height=350,
                              yaxis_title="P&L ($)")
    st.plotly_chart(fig_stress, use_container_width=True)
    st.dataframe(stress_df[["Scenario", "Move", "P&L"]], use_container_width=True)
