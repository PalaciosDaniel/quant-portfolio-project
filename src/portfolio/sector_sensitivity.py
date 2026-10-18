import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# --- Portfolio construction --------------------------------------------------
MODEL_LABEL = "Random Forest"
SIGNAL_WEIGHTING_K = 1.0
MAX_POSITION_WEIGHT = 0.05
MIN_EFFECTIVE_WEIGHT = 0.005
MAX_TURNOVER_DESIGN = 0.25
REBALANCING_DAYS_FIXED = 21  # fixed to isolate the horizon effect (see 09 §5)
BASE_TRANSACTION_COST = 0.0015  # 15 bps, same as notebook 09
OOS_END = pd.Timestamp("2026-06-18")

from src.portfolio.hyperparameter_sensitivity import (
    apply_max_position_weight,
    apply_min_effective_weight,
    compute_net_returns,
)
from src.portfolio.metrics import calculate_performance_metrics
from src.portfolio.constraints import apply_turnover_constraint_with_drift

# NOTE: assumed available in the notebook namespace: `frozen_tickers`,
# `simple_returns_ext`, `TRADING_DAYS_PER_YEAR` and
# `apply_turnover_constraint_with_drift`.

# --- Experiment design -------------------------------------------------------
SECTORS = ["Technology", "Industrials", "Financial Services", "Healthcare"]
CUTS = {"top_10": 1, "top_20": 2, "top_30": 3}  # number of deciles included
PRIMARY_CUT = "top_20"  # fixed before looking at performance
N_REF = 50  # names in the main portfolio (10% of 496)
FINAL_PORTFOLIO = "long_only_top_10_signal_weight"
OOS_PRED_START = "2025-01-16"

# --- Paths -------------------------------------------------------------------
SECTORS_PATH = Path("../data/raw/ticker_sectors.csv")
FINAL_WEIGHTS_PATH = "../data/portfolio_results/final_portfolio_weights.parquet"
OOS_PREDICTIONS_PATH = "../data/model_results/final_model/oos/oos_predictions.parquet"

# --- Yahoo Finance download --------------------------------------------------
YAHOO_ALIAS = {}  # tickers Yahoo does not recognize under the same symbol
MAX_RETRIES = 3
SLEEP_BETWEEN_CALLS = 0.3
SLEEP_ON_RETRY = 3.0
SAVE_EVERY = 100
MANUAL_SECTORS = {  # tickers Yahoo fails to resolve
    "FISV": {
        "yahoo_symbol": "FISV",
        "sector": "Technology",
        "industry": "Information Technology Services",
    },
}

def fetch_sector_info(ticker):
    """Fetch sector and industry for one ticker, with retries."""
    yahoo_symbol = YAHOO_ALIAS.get(ticker, ticker).replace(".", "-")
    for attempt in range(MAX_RETRIES):
        try:
            info = yf.Ticker(yahoo_symbol).info
            sector = info.get("sector")
            if sector:
                return {
                    "ticker": ticker,
                    "yahoo_symbol": yahoo_symbol,
                    "sector": sector,
                    "industry": info.get("industry"),
                }
        except Exception as e:
            print(f"  {ticker}: error on attempt {attempt + 1} -> {e}")
        time.sleep(SLEEP_ON_RETRY)
    return {"ticker": ticker, "yahoo_symbol": yahoo_symbol, "sector": None, "industry": None}


def download_sectors(tickers, output_path, save_every=SAVE_EVERY):
    """Download sectors for all tickers, resuming from saved results."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        saved = pd.read_csv(output_path)
        saved = saved[saved["sector"].notna()]
        done, records = set(saved["ticker"]), saved.to_dict("records")
    else:
        done, records = set(), []

    pending = [t for t in tickers if t not in done]
    print(f"Pending: {len(pending)} of {len(tickers)}")

    for i, ticker in enumerate(pending, start=1):
        records.append(fetch_sector_info(ticker))
        time.sleep(SLEEP_BETWEEN_CALLS)
        if i % save_every == 0:
            pd.DataFrame(records).to_csv(output_path, index=False)
            print(f"  {i}/{len(pending)} processed")

    sectors_df = (
        pd.DataFrame(records)
        .drop_duplicates(subset="ticker", keep="last")
        .set_index("ticker")
        .reindex(tickers)
        .reset_index()
    )
    sectors_df.to_csv(output_path, index=False)
    return sectors_df


def patch_manual_sectors(path, overrides):
    """Overwrite (or append) sector rows for tickers Yahoo cannot resolve."""
    df = pd.read_csv(path)
    for ticker, fields in overrides.items():
        mask = df["ticker"] == ticker
        if mask.any():
            df.loc[mask, list(fields)] = list(fields.values())
        else:
            df = pd.concat([df, pd.DataFrame([{"ticker": ticker, **fields}])], ignore_index=True)
    df.to_csv(path, index=False)
    return df


def report_sector_coverage(sectors_df):
    """Print how many tickers have a sector, which are missing, and the sector counts."""
    missing = sectors_df.loc[sectors_df["sector"].isna(), "ticker"].tolist()
    print(f"Tickers with sector: {sectors_df['sector'].notna().sum()} / {len(sectors_df)}")
    print(f"Missing ({len(missing)}): {missing}")
    print("\nTickers per sector:")
    print(sectors_df["sector"].value_counts(dropna=False))


def load_final_holdings(weights, sector_map, portfolio=FINAL_PORTFOLIO, model=MODEL_LABEL):
    """Positive-weight holdings of the final portfolio, tagged with their sector."""
    holdings = weights.loc[
        (weights["portfolio"] == portfolio)
        & (weights["model"] == model)
        & (weights["weight"] > 0),
        ["date", "ticker", "weight"],
    ].copy()
    holdings["sector"] = holdings["ticker"].map(sector_map)
    assert holdings["sector"].notna().all(), "Tickers without sector in holdings"
    return holdings


def load_global_rank(path, sector_map, final_weights, start=OOS_PRED_START, end=OOS_END):
    """Global OOS rank signal, clipped to the final portfolio window, plus each row's sector."""
    preds = pd.read_parquet(path)
    d = preds.index.get_level_values(0)
    rank = preds.loc[(d >= start) & (d <= end), "prediction_rf_rank"]

    d = rank.index.get_level_values("date")
    rank = rank[(d >= final_weights["date"].min()) & (d <= final_weights["date"].max())]

    sector_of = rank.index.get_level_values("ticker").map(sector_map)
    assert not pd.isna(sector_of).any(), "Tickers without sector in predictions"
    return rank, sector_of


def sector_exposure_summary(holdings, sector_map):
    """Average names and weight per sector (vs. universe share), zeros included."""
    by_sector_date = (
        holdings.groupby(["date", "sector"])
        .agg(n_names=("ticker", "nunique"), weight=("weight", "sum"))
        .reset_index()
    )
    n_dates = holdings["date"].nunique()
    summary = pd.DataFrame(
        {
            "avg_names": by_sector_date.groupby("sector")["n_names"].sum() / n_dates,
            "avg_weight": by_sector_date.groupby("sector")["weight"].sum() / n_dates,
        }
    )
    summary["universe_share"] = sector_map.value_counts(normalize=True)
    return summary.sort_values("avg_weight", ascending=False)


def sector_holdings_stats(holdings, sectors):
    """Number of names and weight per sector across dates (zero-filled)."""
    all_dates = holdings["date"].drop_duplicates().sort_values()
    rows = {}
    for sector in sectors:
        sub = holdings[holdings["sector"] == sector]
        n = sub.groupby("date")["ticker"].nunique().reindex(all_dates, fill_value=0)
        w = sub.groupby("date")["weight"].sum().reindex(all_dates, fill_value=0.0)
        rows[sector] = {
            "n_min": n.min(),
            "n_median": n.median(),
            "n_max": n.max(),
            "days_with_zero": int((n == 0).sum()),
            "weight_mean": w.mean(),
        }
    return pd.DataFrame(rows).T.round(3)


def audit_sector_universe(global_rank, sector_of, sectors, cuts):
    """Names per date (min/median/max) for each sector and cut, plus the minimum feasible cap."""
    audit_rows = {}
    for sector in sectors:
        sub = global_rank[sector_of == sector]
        rank_in = sub.groupby(level="date").rank(method="first", pct=True)

        counts = {"n_universe": sub.groupby(level="date").size()}
        for name, n_deciles in cuts.items():
            counts[name] = (rank_in > 1 - n_deciles / 10).groupby(level="date").sum()
        audit_rows[sector] = pd.DataFrame(counts).agg(["min", "median", "max"]).T

    audit = pd.concat(audit_rows, names=["sector", "metric"])
    feasible = pd.DataFrame(
        {s: {cut: 1 / audit_rows[s].loc[cut, "min"] for cut in cuts} for s in sectors}
    ).T.round(3)
    return audit, feasible


def build_sector_weights(predictions, label, n_deciles):
    """Same logic as build_top_decile_weights (§4), with a variable number of deciles."""
    pct_rank = predictions.groupby(level="date").rank(method="first", pct=True)
    decile = np.ceil(pct_rank * 10).clip(upper=10).astype("Int64")

    selected = predictions.loc[decile >= 11 - n_deciles]
    selected_rank = selected.groupby(level="date").rank(method="first", pct=True)

    strength = selected_rank ** SIGNAL_WEIGHTING_K
    weights = strength / strength.groupby(level="date").transform("sum")

    out = weights.rename("weight").reset_index()
    out["model"] = MODEL_LABEL
    out["portfolio"] = label
    return out[["date", "ticker", "model", "portfolio", "weight"]]


def apply_constraints_custom(all_weights, max_weight, min_weight):
    """Same as apply_weight_constraints (§4), with the limits as arguments."""
    constrained = []
    for (model, portfolio, date), group in all_weights.groupby(
        ["model", "portfolio", "date"], sort=False
    ):
        w = apply_max_position_weight(group["weight"].to_numpy(), max_weight=max_weight)
        w = apply_min_effective_weight(w, min_weight=min_weight)
        constrained.append(
            pd.DataFrame(
                {
                    "date": date,
                    "ticker": group["ticker"].to_numpy(),
                    "model": model,
                    "portfolio": portfolio,
                    "weight": w,
                }
            )
        )
    return pd.concat(constrained, ignore_index=True)


def sector_max_weight(n_selected_min):
    """Cap keeping the main portfolio's slack relative to the natural maximum 2/(N+1)."""
    natural_ref = 2 / (N_REF + 1)
    natural_sector = 2 / (n_selected_min + 1)
    return MAX_POSITION_WEIGHT * natural_sector / natural_ref


def sector_min_weight(n_selected_min):
    """Minimum weight keeping the main portfolio's proportion to equal weight."""
    return (MIN_EFFECTIVE_WEIGHT * N_REF) / n_selected_min


def slug(sector):
    return sector.lower().replace(" ", "_")


def build_sector_portfolios(global_rank, sector_of, sectors, cuts, simple_returns):
    """Raw -> constrained -> turnover-constrained portfolios for every (sector, cut)."""
    results = {}
    for sector in sectors:
        sector_pred = global_rank[sector_of == sector]

        for cut_name, n_deciles in cuts.items():
            label = f"{slug(sector)}_{cut_name}"

            raw = build_sector_weights(sector_pred, label, n_deciles)
            n_min = raw.groupby("date")["ticker"].nunique().min()
            max_w = sector_max_weight(n_min)
            min_w = sector_min_weight(n_min)
            assert max_w >= 1 / n_min, f"{label}: max_weight {max_w:.3f} < 1/N"

            constrained = apply_constraints_custom(raw, max_weight=max_w, min_weight=min_w)
            rebalance_w, daily_w = apply_turnover_constraint_with_drift(
                portfolio_weights=constrained,
                simple_returns=simple_returns,
                rebalancing_days=REBALANCING_DAYS_FIXED,
                max_turnover_design=MAX_TURNOVER_DESIGN,
                min_weight=MIN_EFFECTIVE_WEIGHT,
                max_weight=max_w,
            )

            results[(sector, cut_name)] = {
                "raw": raw,
                "constrained": constrained,
                "rebalance": rebalance_w,
                "daily": daily_w,
                "n_min": n_min,
                "max_weight": max_w,
            }
            print(f"{label}: N_min={n_min}, max_weight={max_w:.3f}")
    return results


def build_sector_benchmark_weights(rank_series, sector, label, sector_map):
    """Equal-weight benchmark over the full within-sector universe."""
    tickers = rank_series.index.get_level_values("ticker")
    sub = rank_series[sector_map.reindex(tickers).to_numpy() == sector]
    n = sub.groupby(level="date").transform("size")
    out = (1 / n).rename("weight").reset_index()[["date", "ticker", "weight"]]
    out["model"] = MODEL_LABEL
    out["portfolio"] = label
    return out[["date", "ticker", "model", "portfolio", "weight"]]


def rebalance_dates(df, oos_end):
    d = pd.to_datetime(df["date"].unique())
    return sorted(x for x in d if x <= oos_end)


def at_rebalance(df, dates, suffix=""):
    """Keep only rows on the given rebalancing dates; optionally tag the portfolio name."""
    out = df[pd.to_datetime(df["date"]).isin(dates)].copy()
    if suffix:
        out["portfolio"] = out["portfolio"] + suffix
    return out


def assemble_rebalance_weights(sector_results, benchmarks, oos_end):
    """Stack constrained, no-turnover and benchmark weights on a common calendar."""
    ref_dates = rebalance_dates(next(iter(sector_results.values()))["rebalance"], oos_end)
    for key, res in sector_results.items():
        assert rebalance_dates(res["rebalance"], oos_end) == ref_dates, f"Different calendar in {key}"

    frames = []
    for res in sector_results.values():
        frames.append(at_rebalance(res["rebalance"], ref_dates))
        frames.append(at_rebalance(res["constrained"], ref_dates, suffix="_noturn"))
    for bench in benchmarks.values():
        frames.append(at_rebalance(bench, ref_dates))

    all_rebalance = pd.concat(frames, ignore_index=True)

    bench_sum = (
        all_rebalance[all_rebalance["portfolio"].str.endswith("equal_weight")]
        .groupby(["portfolio", "date"])["weight"]
        .sum()
    )
    assert np.allclose(bench_sum, 1.0), "Benchmark weights do not sum to 1"
    return all_rebalance


def label_portfolios(perf, sectors, cuts):
    """Tag each portfolio with its sector, cut and version (benchmark / turnover / no turnover)."""
    meta = {}
    for sector in sectors:
        meta[f"{slug(sector)}_equal_weight"] = (sector, "benchmark", "equal_weight")
        for cut in cuts:
            meta[f"{slug(sector)}_{cut}"] = (sector, cut, "with_turnover")
            meta[f"{slug(sector)}_{cut}_noturn"] = (sector, cut, "no_turnover")

    perf = perf.copy()
    perf[["sector", "cut", "version"]] = perf["portfolio"].map(meta).apply(pd.Series)
    return perf


def compare_to_benchmark(perf):
    """Attach each sector's benchmark metrics and compute excess CAGR / Sharpe delta."""
    bench = (
        perf[perf["cut"] == "benchmark"]
        .set_index("sector")[["CAGR", "Sharpe", "Sortino", "Calmar", "max_DD"]]
        .add_prefix("bench_")
    )
    sector_perf = perf[perf["cut"] != "benchmark"].join(bench, on="sector")
    sector_perf["excess_CAGR"] = sector_perf["CAGR"] - sector_perf["bench_CAGR"]
    sector_perf["delta_Sharpe"] = sector_perf["Sharpe"] - sector_perf["bench_Sharpe"]
    return sector_perf


def add_active_stats(sector_perf, returns_wide, trading_days):
    """Tracking error and information ratio on daily active returns vs. the sector benchmark."""

    def active_stats(row):
        active = returns_wide[row["portfolio"]] - returns_wide[f"{slug(row['sector'])}_equal_weight"]
        te = active.std(ddof=1) * np.sqrt(trading_days)
        ir = active.mean() * trading_days / te
        return pd.Series({"TE": te, "IR": ir})

    sector_perf = sector_perf.copy()
    sector_perf[["TE", "IR"]] = sector_perf.apply(active_stats, axis=1)
    return sector_perf


def format_sector_table(sector_perf, cut=PRIMARY_CUT, version="with_turnover"):
    """Formatted comparison table for one cut and version."""
    cols = [
        "sector", "cut", "CAGR", "bench_CAGR", "excess_CAGR",
        "Sharpe", "bench_Sharpe", "delta_Sharpe", "Sortino", "Calmar",
        "max_DD", "bench_max_DD", "TE", "IR",
    ]
    pct_cols = ["CAGR", "bench_CAGR", "excess_CAGR", "max_DD", "bench_max_DD", "TE"]
    ratio_cols = ["Sharpe", "bench_Sharpe", "delta_Sharpe", "Sortino", "Calmar", "IR"]

    display = (
        sector_perf[(sector_perf["version"] == version) & (sector_perf["cut"] == cut)][cols]
        .sort_values("sector")
    )
    return display.to_string(
        index=False,
        formatters={
            **{c: (lambda x: f"{x:.2%}") for c in pct_cols},
            **{c: (lambda x: f"{x:.3f}") for c in ratio_cols},
        },
    )