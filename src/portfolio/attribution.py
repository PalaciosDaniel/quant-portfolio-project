import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib.ticker import PercentFormatter

# --- Rolling CAPM ------------------------------------------------------------
ROLLING_WINDOW = 60

# --- Stress resampling -------------------------------------------------------
N_BOOTSTRAP = 2000
BLOCK_SIZE = 21              # matches the holding period
SYNTHETIC_YEAR_BLOCKS = 12   # 12 blocks x 21 days = 252 trading days
STRESS_PERCENTILE = 33       # keep the worst third of benchmark blocks
BOOTSTRAP_SEED = 42
TRADING_DAYS_PER_YEAR = 252

def calculate_capture_ratio_daily(s_strat, s_bmk):
    """Up/down capture ratios based on daily average returns."""
    up = s_bmk > 0
    down = s_bmk <= 0

    bmk_up_mean, strat_up_mean = s_bmk[up].mean(), s_strat[up].mean()
    bmk_down_mean, strat_down_mean = s_bmk[down].mean(), s_strat[down].mean()

    up_capture = strat_up_mean / bmk_up_mean if bmk_up_mean > 0 else np.nan
    down_capture = strat_down_mean / bmk_down_mean if bmk_down_mean < 0 else np.nan
    spread = (
        up_capture - down_capture
        if (pd.notna(up_capture) and pd.notna(down_capture))
        else np.nan
    )

    return {
        "Up_Days_Count": up.sum(),
        "Down_Days_Count": down.sum(),
        "BMK_Up_Mean": bmk_up_mean,
        "Strat_Up_Mean": strat_up_mean,
        "Up_Capture": up_capture,
        "BMK_Down_Mean": bmk_down_mean,
        "Strat_Down_Mean": strat_down_mean,
        "Down_Capture": down_capture,
        "Capture_Spread": spread,
    }


def print_capture_summary(capture):
    n_up, n_down = capture["Up_Days_Count"], capture["Down_Days_Count"]
    table = pd.DataFrame(
        [
            {
                "Metric": "Up-Market Capture",
                "Benchmark Avg Return": f"{capture['BMK_Up_Mean']:.3%}",
                "Strategy Avg Return": f"{capture['Strat_Up_Mean']:.3%}",
                "Capture Ratio": f"{capture['Up_Capture']:.1%}",
            },
            {
                "Metric": "Down-Market Capture",
                "Benchmark Avg Return": f"{capture['BMK_Down_Mean']:.3%}",
                "Strategy Avg Return": f"{capture['Strat_Down_Mean']:.3%}",
                "Capture Ratio": f"{capture['Down_Capture']:.1%}",
            },
        ]
    )

    print("=" * 90)
    print("UP/DOWN MARKET CAPTURE RATIOS (DAILY AVERAGE METHOD)")
    print("=" * 90)
    print(f"Total Days Analyzed: {n_up + n_down} ({n_up} Up / {n_down} Down)")
    print("-" * 90)
    print(table.to_string(index=False))
    print("-" * 90)
    print(f"Capture Spread (Asymmetry): {capture['Capture_Spread']:.1%} (Higher is better)")
    print("=" * 90)


def plot_capture_ratios(capture):
    """Bar chart of up- and down-market capture ratios."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = ["Up-Market\nCapture", "Down-Market\nCapture"]
    values = [capture["Up_Capture"] * 100, capture["Down_Capture"] * 100]
    bars = ax.bar(labels, values, color=["#2ca02c", "#d62728"], width=0.5)

    ax.axhline(100, color="gray", linestyle="--", alpha=0.7, label="100% (Market Beta = 1.0)")
    ax.axhline(0, color="black", linewidth=1)

    for bar in bars:
        y = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y + (5 if y > 0 else -10),
            f"{y:.1f}%",
            ha="center",
            va="bottom" if y > 0 else "top",
            fontweight="bold",
        )

    ax.set_title("Strategy vs S&P 500: Market Capture Ratios", fontsize=14, pad=15)
    ax.set_ylabel("Capture Ratio (%)", fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.show()


def rolling_capm_tracking_error(
    strategy, benchmark, window=ROLLING_WINDOW, trading_days=TRADING_DAYS_PER_YEAR
):
    """Rolling OLS beta, annualized alpha and annualized tracking error."""
    df = pd.DataFrame({"strategy": strategy, "benchmark": benchmark}).dropna()

    records = []
    for i in range(window, len(df)):
        w = df.iloc[i - window : i]
        y, x = w["strategy"].values, w["benchmark"].values

        fit = sm.OLS(y, sm.add_constant(x)).fit()
        alpha_daily, beta = fit.params
        tracking_error = np.std(y - x, ddof=1) * np.sqrt(trading_days)

        records.append(
            {
                "date": df.index[i - 1],  # last date of the window
                "Rolling_Beta": beta,
                "Rolling_Alpha_Ann": alpha_daily * trading_days,
                "Rolling_TE_Ann": tracking_error,
            }
        )
    return pd.DataFrame(records).set_index("date")


def plot_rolling_risk(df, window=ROLLING_WINDOW):
    """Three-panel dashboard: rolling beta, alpha and tracking error."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Beta
    axes[0].plot(df.index, df["Rolling_Beta"], color="purple", linewidth=2)
    axes[0].axhline(1.0, color="black", linestyle="--", alpha=0.5, label="Market Beta = 1.0")
    axes[0].axhline(df["Rolling_Beta"].mean(), color="purple", linestyle=":", alpha=0.5, label="Mean Beta")
    axes[0].set_title(f"Rolling Beta ({window}-Day Window)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Beta")
    axes[0].legend(loc="upper left")

    # Alpha (green above zero, red below)
    alpha = df["Rolling_Alpha_Ann"]
    axes[1].plot(df.index, alpha, color="green", linewidth=2)
    axes[1].fill_between(df.index, 0, alpha, where=(alpha >= 0), color="green", alpha=0.3)
    axes[1].fill_between(df.index, 0, alpha, where=(alpha < 0), color="red", alpha=0.3)
    axes[1].axhline(0, color="black", linestyle="-", linewidth=1)
    axes[1].set_title(f"Rolling Annualized Alpha ({window}-Day Window)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Alpha")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))

    # Tracking error
    axes[2].plot(df.index, df["Rolling_TE_Ann"], color="darkorange", linewidth=2)
    axes[2].fill_between(df.index, 0, df["Rolling_TE_Ann"], color="darkorange", alpha=0.1)
    axes[2].set_title(f"Rolling Tracking Error (Annualized, {window}-Day Window)",
                      fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Tracking Error")
    axes[2].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))

    plt.suptitle("Dynamic Risk & Attribution Profile (Random Forest Top 10)", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()


def print_rolling_summary(df, window=ROLLING_WINDOW):
    print("=" * 90)
    print(f"DYNAMIC RISK SUMMARY (Rolling {window} Days)")
    print("=" * 90)
    print(df.describe().T[["mean", "min", "max", "std"]].round(3))
    print("=" * 90)


def build_stress_pool(benchmark, block_size=BLOCK_SIZE, percentile=STRESS_PERCENTILE):
    """Start positions of the worst benchmark blocks (compounded block return)."""
    valid_starts = np.arange(len(benchmark) - block_size + 1)
    block_returns = np.array(
        [(1.0 + benchmark.iloc[s : s + block_size]).prod() - 1.0 for s in valid_starts]
    )
    threshold = np.percentile(block_returns, percentile)
    return valid_starts, valid_starts[block_returns <= threshold], threshold


def simulate_stress_years(
    strategy, benchmark, stressed_starts,
    n_bootstrap=N_BOOTSTRAP, block_size=BLOCK_SIZE,
    n_blocks=SYNTHETIC_YEAR_BLOCKS, seed=BOOTSTRAP_SEED,
):
    """Synthetic years of chained stress blocks; returns (strategy, benchmark) total returns."""
    np.random.seed(seed)
    strat_returns, bmk_returns = [], []

    for _ in range(n_bootstrap):
        starts = np.random.choice(stressed_starts, size=n_blocks, replace=True)
        strat = np.concatenate([strategy.iloc[s : s + block_size].values for s in starts])
        bmk = np.concatenate([benchmark.iloc[s : s + block_size].values for s in starts])
        strat_returns.append((1.0 + strat).prod() - 1.0)
        bmk_returns.append((1.0 + bmk).prod() - 1.0)

    return np.array(strat_returns), np.array(bmk_returns)


def stress_results_table(strat_returns, bmk_returns, strategy_label):
    """Mean, 95% interval and profitability rate of the synthetic years, plus the win rate."""

    def row(label, r):
        low, high = np.percentile(r, [2.5, 97.5])
        return {
            "Portfolio": label,
            "Mean Synthetic CAGR": r.mean(),
            "95% CI Low": low,
            "95% CI High": high,
            "% of Years Profitable": (r > 0).mean(),
        }

    table = pd.DataFrame(
        [row("S&P 500 (Benchmark)", bmk_returns), row(f"{strategy_label} (Strategy)", strat_returns)]
    )
    win_rate = (strat_returns > bmk_returns).mean()
    return table, win_rate


def print_stress_results(table, win_rate, n_blocks, n_stressed, threshold):
    print("=" * 100)
    print(f"CONDITIONAL BOOTSTRAP: {N_BOOTSTRAP} SYNTHETIC STRESS YEARS")
    print("=" * 100)
    print(f"Total overlapping blocks available: {n_blocks}")
    print(f"Stressed blocks pool (worst {STRESS_PERCENTILE}%): {n_stressed}")
    print(f"Stress threshold (max S&P 500 return in these blocks): {threshold:.2%}")
    print("-" * 100)
    print(
        table.to_string(
            index=False,
            formatters={
                "Mean Synthetic CAGR": lambda x: f"{x:+.2%}",
                "95% CI Low": lambda x: f"{x:+.2%}",
                "95% CI High": lambda x: f"{x:+.2%}",
                "% of Years Profitable": lambda x: f"{x:.1%}",
            },
        )
    )
    print("-" * 100)
    print(f"Strategy Outperformance Rate (beats benchmark in stress): {win_rate:.1%}")
    print("=" * 100)


def active_drawdown_analysis(strategy, benchmark):
    """Relative wealth index, active drawdown and longest underwater stretch."""
    relative_wealth = (1.0 + strategy).cumprod() / (1.0 + benchmark).cumprod()
    running_max = relative_wealth.cummax()
    active_dd = relative_wealth / running_max - 1.0

    is_underwater = active_dd < 0
    durations = is_underwater.groupby((~is_underwater).cumsum()).sum()

    return {
        "relative_wealth": relative_wealth,
        "running_max": running_max,
        "active_drawdown": active_dd,
        "max_active_dd": active_dd.min(),
        "max_underwater_days": int(durations.max()),
    }


def print_active_drawdown_summary(result):
    print("=" * 90)
    print("RELATIVE DRAWDOWN ANALYSIS (VS S&P 500)")
    print("=" * 90)
    print(f"Maximum Active Drawdown (peak-to-trough underperformance): {result['max_active_dd']:.2%}")
    print(f"Maximum Underwater Duration (longest stretch lagging S&P): {result['max_underwater_days']} trading days")
    print("=" * 90)


def plot_active_drawdown(result):
    """Relative wealth index (top) and active drawdown (bottom)."""
    rel, run_max, dd = result["relative_wealth"], result["running_max"], result["active_drawdown"]

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})

    axes[0].plot(rel.index, rel, color="blue", linewidth=2, label="Relative Wealth (Strategy / S&P 500)")
    axes[0].plot(run_max.index, run_max, color="gray", linestyle="--", alpha=0.7, label="Running Maximum")
    axes[0].set_title("Relative Wealth Index (Outperformance vs Benchmark)", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Ratio (Base 1.0)")
    axes[0].legend(loc="upper left")

    axes[1].fill_between(dd.index, dd, 0, color="red", alpha=0.4)
    axes[1].plot(dd.index, dd, color="darkred", linewidth=1.5)
    axes[1].set_title("Active Drawdown (Underperformance vs S&P 500)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Active Drawdown (%)")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))

    plt.tight_layout()
    plt.show()