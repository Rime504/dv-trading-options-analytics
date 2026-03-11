"""
Volatility Surface Module
- SVI (Stochastic Volatility Inspired) smile parametrization
- SABR smile fitting
- Implied vol surface construction from options chain
- <5s render time for 3D interactive surface
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import RectBivariateSpline, griddata
from typing import Optional, Tuple, List, Dict
from black_scholes import implied_volatility


# ---------------------------------------------------------------------------
# SVI Parametrization  (Gatheral 2004)
# w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))
# where k = log(K/F), w = total implied variance = sigma_imp^2 * T
# ---------------------------------------------------------------------------

def svi_raw(k: np.ndarray, a: float, b: float, rho: float,
            m: float, sigma: float) -> np.ndarray:
    """SVI raw parametrization: total variance w(k)."""
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def svi_implied_vol(k: np.ndarray, T: float, a: float, b: float,
                    rho: float, m: float, sigma: float) -> np.ndarray:
    """Convert SVI total variance to implied vol."""
    w = svi_raw(k, a, b, rho, m, sigma)
    w = np.maximum(w, 1e-10)
    return np.sqrt(w / T)


def fit_svi_slice(
    strikes: np.ndarray,
    ivols: np.ndarray,
    F: float,
    T: float,
    method: str = "L-BFGS-B",
) -> Tuple[np.ndarray, float]:
    """
    Fit SVI parameters [a, b, rho, m, sigma] to a single expiry slice.
    Returns (params, rmse).
    """
    k = np.log(strikes / F)
    w_market = (ivols ** 2) * T

    def objective(params):
        a, b, rho, m, sig = params
        w_model = svi_raw(k, a, b, rho, m, sig)
        return np.sum((w_model - w_market) ** 2)

    def no_arb_constraints(params):
        """Butterfly no-arbitrage: b*(1+|rho|) <= 4 and a >= -b*sig."""
        a, b, rho, m, sig = params
        return [4.0 - b * (1 + abs(rho)), a + b * sig]

    bounds = [
        (-1.0, 1.0),   # a: total variance intercept
        (0.001, 2.0),  # b: slope
        (-0.999, 0.999),  # rho: correlation
        (-1.0, 1.0),   # m: ATM shift
        (0.001, 2.0),  # sigma: minimum variance
    ]

    # Initial guess: flat vol
    avg_w = np.mean(w_market)
    x0 = [avg_w, 0.1, -0.3, 0.0, 0.2]

    constraints = [{"type": "ineq", "fun": no_arb_constraints}]

    try:
        res = minimize(objective, x0, method=method, bounds=bounds,
                       constraints=constraints if method != "L-BFGS-B" else None,
                       options={"maxiter": 1000, "ftol": 1e-12})
        params = res.x
    except Exception:
        params = np.array(x0)

    w_fit = svi_raw(k, *params)
    rmse = np.sqrt(np.mean((np.sqrt(np.maximum(w_fit / T, 1e-10)) - ivols) ** 2))
    return params, rmse


# ---------------------------------------------------------------------------
# SABR Model (Hagan et al. 2002)
# ---------------------------------------------------------------------------

def sabr_implied_vol(F: float, K: float, T: float,
                     alpha: float, beta: float, rho: float, nu: float) -> float:
    """SABR implied volatility approximation (Hagan 2002)."""
    if abs(F - K) < 1e-10:
        # ATM formula
        fmid = F ** (1 - beta)
        term1 = alpha / fmid
        term2 = (1 + ((1 - beta) ** 2 / 24 * alpha ** 2 / fmid ** 2
                      + 0.25 * rho * beta * nu * alpha / fmid
                      + (2 - 3 * rho ** 2) / 24 * nu ** 2) * T)
        return term1 * term2

    log_FK = np.log(F / K)
    FK_mid = np.sqrt(F * K)
    fmid = FK_mid ** (1 - beta)

    z = nu / alpha * fmid * log_FK
    x_z = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))

    if abs(x_z) < 1e-10:
        x_z = 1e-10

    A = alpha / (fmid * (1 + (1 - beta) ** 2 / 24 * log_FK ** 2
                          + (1 - beta) ** 4 / 1920 * log_FK ** 4))
    B = z / x_z
    C = (1 + ((1 - beta) ** 2 / 24 * alpha ** 2 / fmid ** 2
              + 0.25 * rho * beta * nu * alpha / fmid
              + (2 - 3 * rho ** 2) / 24 * nu ** 2) * T)

    return A * B * C


def fit_sabr(strikes: np.ndarray, ivols: np.ndarray,
             F: float, T: float, beta: float = 0.5) -> Tuple[np.ndarray, float]:
    """Fit SABR [alpha, rho, nu] given fixed beta."""
    def objective(params):
        alpha, rho, nu = params
        if alpha <= 0 or nu <= 0 or abs(rho) >= 1:
            return 1e10
        model_vols = np.array([
            sabr_implied_vol(F, K, T, alpha, beta, rho, nu) for K in strikes
        ])
        return np.sum((model_vols - ivols) ** 2)

    x0 = [0.3, -0.3, 0.4]
    bounds = [(0.001, 5.0), (-0.999, 0.999), (0.001, 5.0)]
    res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
    alpha, rho, nu = res.x

    model_vols = np.array([sabr_implied_vol(F, K, T, alpha, beta, rho, nu) for K in strikes])
    rmse = np.sqrt(np.mean((model_vols - ivols) ** 2))
    return res.x, rmse


# ---------------------------------------------------------------------------
# Full Volatility Surface
# ---------------------------------------------------------------------------

class VolSurface:
    """
    Construct, fit, and interpolate a full implied vol surface.
    Inputs: options chain data with strikes, expiries, and market prices.
    """

    def __init__(self, S: float, r: float, q: float = 0.0):
        self.S = S
        self.r = r
        self.q = q
        self.svi_params: Dict[float, np.ndarray] = {}  # T -> SVI params
        self.surface_grid: Optional[np.ndarray] = None
        self.K_grid: Optional[np.ndarray] = None
        self.T_grid: Optional[np.ndarray] = None

    def build_from_chain(
        self,
        chain: List[Dict],
        option_type: str = "call",
    ) -> None:
        """
        Build surface from options chain.
        chain: list of dicts with keys: K, T, mid_price
        """
        from collections import defaultdict
        slices = defaultdict(list)
        for row in chain:
            slices[row["T"]].append(row)

        for T, rows in sorted(slices.items()):
            if T <= 0:
                continue
            F = self.S * np.exp((self.r - self.q) * T)
            strikes = np.array([r["K"] for r in rows])
            prices = np.array([r["mid_price"] for r in rows])

            # Compute implied vols
            ivols = np.array([
                implied_volatility(p, self.S, K, T, self.r, self.q, option_type)
                for K, p in zip(strikes, prices)
            ])

            # Filter valid ivols
            mask = np.isfinite(ivols) & (ivols > 0.01) & (ivols < 5.0)
            if mask.sum() < 3:
                continue

            strikes_f, ivols_f = strikes[mask], ivols[mask]
            params, _ = fit_svi_slice(strikes_f, ivols_f, F, T)
            self.svi_params[T] = params

        self._build_grid()

    def build_synthetic(
        self,
        K_range: Tuple[float, float] = (0.7, 1.3),
        T_range: Tuple[float, float] = (0.05, 2.0),
        n_strikes: int = 50,
        n_expiries: int = 20,
        atm_vol: float = 0.20,
        skew: float = -0.1,
        smile: float = 0.05,
    ) -> None:
        """Build a parametric synthetic vol surface for demo/testing."""
        K_norm = np.linspace(K_range[0], K_range[1], n_strikes)
        T_vals = np.linspace(T_range[0], T_range[1], n_expiries)
        K_abs = K_norm * self.S

        self.K_grid = K_abs
        self.T_grid = T_vals
        surface = np.zeros((n_strikes, n_expiries))

        for j, T in enumerate(T_vals):
            F = self.S * np.exp((self.r - self.q) * T)
            k = np.log(K_abs / F)
            # Simple quadratic smile
            iv = atm_vol + skew * k + smile * k ** 2
            # Term structure: short-dated more vol
            iv *= (1 + 0.05 / np.sqrt(T))
            surface[:, j] = np.maximum(iv, 0.01)

        self.surface_grid = surface

    def _build_grid(self, n_K: int = 50, n_T: int = 20) -> None:
        """Interpolate fitted SVI slices onto a regular grid."""
        if not self.svi_params:
            return
        T_vals = np.array(sorted(self.svi_params.keys()))
        K_range = (self.S * 0.6, self.S * 1.4)
        K_abs = np.linspace(K_range[0], K_range[1], n_K)

        surface = np.zeros((n_K, len(T_vals)))
        for j, T in enumerate(T_vals):
            F = self.S * np.exp((self.r - self.q) * T)
            k = np.log(K_abs / F)
            params = self.svi_params[T]
            surface[:, j] = svi_implied_vol(k, T, *params)

        self.K_grid = K_abs
        self.T_grid = T_vals
        self.surface_grid = surface

    def get_vol(self, K: float, T: float) -> float:
        """Interpolate surface at arbitrary (K, T)."""
        if self.surface_grid is None or self.K_grid is None or self.T_grid is None:
            return 0.20  # default
        if len(self.T_grid) < 2 or len(self.K_grid) < 2:
            return 0.20

        # Bilinear interpolation via griddata
        pts = np.array([[k, t] for k in self.K_grid for t in self.T_grid])
        vals = self.surface_grid.flatten()
        result = griddata(pts, vals, (K, T), method="linear")
        if np.isnan(result):
            result = griddata(pts, vals, (K, T), method="nearest")
        return float(result)

    def local_vol_dupire(self, K: float, T: float, dK: float = 1.0, dT: float = 0.01) -> float:
        """
        Dupire local vol approximation from implied vol surface.
        sigma_loc^2 = (dC/dT) / (0.5*K^2 * d^2C/dK^2)
        Uses finite differences on BS prices.
        """
        from black_scholes import bs_price_vectorized
        iv = self.get_vol(K, T)
        iv_Ku = self.get_vol(K + dK, T)
        iv_Kd = self.get_vol(K - dK, T)
        iv_Tu = self.get_vol(K, T + dT)
        iv_Td = self.get_vol(K, max(T - dT, 1e-4))

        r, q, S = self.r, self.q, self.S

        # Call prices at grid points
        C    = bs_price_vectorized(S, K,      T,      r, iv,    q, "call")
        C_Ku = bs_price_vectorized(S, K + dK, T,      r, iv_Ku, q, "call")
        C_Kd = bs_price_vectorized(S, K - dK, T,      r, iv_Kd, q, "call")
        C_Tu = bs_price_vectorized(S, K,      T + dT, r, iv_Tu, q, "call")
        C_Td = bs_price_vectorized(S, K,      max(T - dT, 1e-4), r, iv_Td, q, "call")

        dC_dT  = (C_Tu - C_Td) / (2 * dT)
        d2C_dK2 = (C_Ku - 2 * C + C_Kd) / (dK ** 2)

        if d2C_dK2 < 1e-12:
            return iv  # fallback

        local_var = dC_dT / (0.5 * K ** 2 * d2C_dK2)
        return float(np.sqrt(max(local_var, 1e-6)))


def term_structure(
    expiries: np.ndarray,
    atm_vols: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Compute term structure metrics from ATM vol curve.
    Returns: forward vols, variance, calendar spread check.
    """
    variances = atm_vols ** 2 * expiries
    fwd_vars = np.diff(variances) / np.diff(expiries)
    fwd_vols = np.sqrt(np.maximum(fwd_vars, 0))

    calendar_arbitrage_free = np.all(np.diff(variances) >= 0)

    return {
        "expiries": expiries,
        "atm_vols": atm_vols,
        "total_variances": variances,
        "forward_vols": fwd_vols,
        "calendar_arb_free": calendar_arbitrage_free,
    }
