"""
Portfolio performance metrics.

Given a sequence of daily portfolio returns (either arithmetic or log),
compute the standard performance statistics used in backtesting:

    - sharpe_ratio:         annualized, risk-adjusted return
    - max_drawdown:         largest peak-to-trough wealth decline
    - annualized_vol:       daily std scaled to annual
    - cagr:                 compound annual growth rate
    - summary:              all of the above as a dict

All functions take a 1-D numpy array of DAILY returns as their primary
input.  By default we assume arithmetic returns (r_t = P_t / P_{t-1} - 1).
If you're passing log returns, set log_returns=True for metrics that care
about compounding (cagr, max_drawdown).
"""

from __future__ import annotations

import numpy as np


TRADING_DAYS_PER_YEAR = 252


# -----------------------------------------------------------------------------
# Individual metrics
# -----------------------------------------------------------------------------

def annualized_return(returns: np.ndarray) -> float:
    """
    Annualized arithmetic mean of daily returns.

        annual_return = mean(daily_returns) * 252

    Useful as the "return" component of the Sharpe ratio.  Does NOT
    account for compounding; use `cagr` for that.
    """
    if returns.size == 0:
        return float("nan")
    return float(returns.mean() * TRADING_DAYS_PER_YEAR)


def annualized_vol(returns: np.ndarray) -> float:
    """
    Annualized volatility (standard deviation) of daily returns.

        annual_vol = std(daily_returns) * sqrt(252)

    Uses sample std with ddof=1 (Bessel-corrected).
    """
    if returns.size < 2:
        return float("nan")
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(returns: np.ndarray, rf_annual: float = 0.0) -> float:
    """
    Annualized Sharpe ratio.

        Sharpe = (annual_return - rf_annual) / annual_vol

    Parameters
    ----------
    returns : 1D array of DAILY arithmetic returns
    rf_annual : annualized risk-free rate (e.g. 0.04 for 4%)

    Returns
    -------
    float
        Annualized Sharpe ratio.  Returns nan if volatility is effectively
        zero (below 1e-10, i.e. below floating-point noise for typical
        daily returns).
    """
    vol = annualized_vol(returns)
    # Use tolerance-based check: any "zero" below 1e-10 annualized vol
    # (i.e. below ~6e-13 daily std) is floating-point noise, not real variation
    if not np.isfinite(vol) or vol < 1e-10:
        return float("nan")
    return (annualized_return(returns) - rf_annual) / vol


def max_drawdown(returns: np.ndarray, log_returns: bool = False) -> float:
    """
    Maximum peak-to-trough decline in cumulative wealth.

    Computes the wealth curve starting from 1, then finds the largest
    percentage drop from any prior peak.  The result is NEGATIVE
    (drawdowns are losses), in the range [-1, 0].

    A drawdown of -0.25 means the portfolio lost 25% from its peak
    at the worst point in the backtest.

    Parameters
    ----------
    returns : 1D array of daily returns
    log_returns : bool
        If True, interpret input as log returns and exponentiate for
        compounding.  If False (default), interpret as arithmetic
        returns and compound as (1 + r).
    """
    if returns.size == 0:
        return float("nan")

    # Build the wealth curve (cumulative product of gross returns, starting at 1)
    if log_returns:
        wealth = np.exp(np.cumsum(returns))
    else:
        wealth = np.cumprod(1.0 + returns)

    # Running maximum of wealth
    running_max = np.maximum.accumulate(wealth)

    # Drawdown at each time: (wealth - peak) / peak
    drawdowns = (wealth - running_max) / running_max

    return float(drawdowns.min())


def cagr(returns: np.ndarray, log_returns: bool = False) -> float:
    """
    Compound annual growth rate.

    Given T days of returns, computes the total return factor and
    annualizes it geometrically:

        total_return = product(1 + r_t)        (arithmetic)
                     = exp(sum(r_t))           (log)
        cagr = total_return^(252/T) - 1

    Returns a fractional growth rate (e.g. 0.08 for 8% per year).
    """
    if returns.size == 0:
        return float("nan")

    T = returns.size
    if log_returns:
        total_return = float(np.exp(returns.sum()))
    else:
        total_return = float(np.prod(1.0 + returns))

    if total_return <= 0.0:
        # Portfolio went bust; CAGR undefined
        return float("nan")

    return total_return ** (TRADING_DAYS_PER_YEAR / T) - 1.0


# -----------------------------------------------------------------------------
# Summary aggregator
# -----------------------------------------------------------------------------

def summary(
    returns: np.ndarray,
    rf_annual: float = 0.0,
    log_returns: bool = False,
) -> dict[str, float]:
    """
    Compute all four standard metrics at once.

    Returns a dict with keys:
        'annual_return', 'annual_vol', 'sharpe', 'max_drawdown', 'cagr'

    This is the function the backtest engine will call to report results
    for each (estimator, rebalance date, bootstrap sample) combination.
    """
    return {
        "annual_return": annualized_return(returns),
        "annual_vol": annualized_vol(returns),
        "sharpe": sharpe_ratio(returns, rf_annual=rf_annual),
        "max_drawdown": max_drawdown(returns, log_returns=log_returns),
        "cagr": cagr(returns, log_returns=log_returns),
    }


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("Metrics smoke test\n" + "=" * 60)

    # --- Analytic check: constant-return series ---
    # Daily return of +0.1% for one trading year should give:
    #   annual_return = 0.001 * 252 = 0.252
    #   annual_vol = 0.0
    #   Sharpe = nan (vol is zero)
    #   max_drawdown = 0.0 (monotonically increasing)
    #   cagr = (1.001)^252 - 1 ≈ 0.2879

    print("\n[Constant +0.1% daily for 252 days]")
    const = np.full(252, 0.001)
    s = summary(const)
    print(f"  annual_return = {s['annual_return']:+.6f}  (expected +0.252)")
    print(f"  annual_vol    = {s['annual_vol']:+.6e}  (expected ~0)")
    print(f"  sharpe        = {s['sharpe']}         (expected nan)")
    print(f"  max_drawdown  = {s['max_drawdown']:+.6f}  (expected 0)")
    print(f"  cagr          = {s['cagr']:+.6f}  (expected +0.287)")

    # --- Symmetric random walk: Sharpe ≈ 0 ---
    print("\n[Zero-mean random returns, 5 years]")
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(252 * 5) * 0.01
    s = summary(noise)
    print(f"  annual_return = {s['annual_return']:+.4f}")
    print(f"  annual_vol    = {s['annual_vol']:+.4f}  (expected ~0.16)")
    print(f"  sharpe        = {s['sharpe']:+.4f}  (expected near 0)")
    print(f"  max_drawdown  = {s['max_drawdown']:+.4f}")
    print(f"  cagr          = {s['cagr']:+.4f}")

    # --- Known drawdown scenario ---
    # Portfolio gains 50% then loses 50% - peak-to-trough drawdown should be -0.5
    print("\n[Up 50%, down 50% scenario]")
    scripted = np.array([0.5, -0.5])
    s = summary(scripted)
    print(f"  max_drawdown  = {s['max_drawdown']:+.4f}  (expected -0.5)")
    print(f"  final wealth  = {np.prod(1 + scripted):.4f}    (expected 0.75)")

    # --- Real S&P 100 equal-weighted backtest ---
    print("\n[Real S&P 100 equal-weighted, full 2014-2024]")
    from src.data_loader import load_returns
    returns_df = load_returns()
    # Equal-weighted portfolio: mean across columns at each day
    port_returns = returns_df.mean(axis=1).to_numpy()  # log returns
    # Convert log -> arithmetic for metrics that want arithmetic
    arith = np.exp(port_returns) - 1.0

    s = summary(arith)
    print(f"  annual_return = {s['annual_return']:+.4f}  ({s['annual_return']*100:+.2f}%)")
    print(f"  annual_vol    = {s['annual_vol']:+.4f}  ({s['annual_vol']*100:+.2f}%)")
    print(f"  sharpe        = {s['sharpe']:+.4f}")
    print(f"  max_drawdown  = {s['max_drawdown']:+.4f}  ({s['max_drawdown']*100:+.2f}%)")
    print(f"  cagr          = {s['cagr']:+.4f}  ({s['cagr']*100:+.2f}%)")