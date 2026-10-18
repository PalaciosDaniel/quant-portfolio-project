import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.features.liquidity import compute_amihud_illiquidity
from src.features.momentum import compute_12_1_momentum
from src.features.volatility import compute_upside_volatility
from src.portfolio.constraints import (
    apply_max_position_weight,
    apply_min_effective_weight,
)
from src.portfolio.metrics import calculate_performance_metrics
from src.preprocessing.factor_preprocessing import (
    compute_forward_return,
    rank_cross_sectional,
    winsorize_cross_sectional,
)

# --- Experiment grid ---------------------------------------------------------
HORIZONS = [5, 10, 21]
WINDOWS = ["expanding", "rolling_5y"]
TRAIN_START = {
    "expanding": pd.Timestamp("2011-01-03"),
    "rolling_5y": pd.Timestamp("2020-01-01"),
}

# --- Sample split ------------------------------------------------------------
TRAIN_END = pd.Timestamp("2024-12-30")
OOS_START = pd.Timestamp("2025-01-15")
OOS_END = pd.Timestamp("2026-06-18")

# --- Features and universe ---------------------------------------------------
FEATURES_RANK = [
    "momentum_12_1_win_rank",
    "upside_volatility_win_rank",
    "log10_amihud_win_rank",
]
FACTOR_COLS = ["momentum_12_1", "upside_volatility", "log10_amihud"]
EXCLUDED_OOS_TICKERS = ["SW", "AMCR"]

# --- Portfolio construction --------------------------------------------------
MODEL_LABEL = "Random Forest"
TOP_DECILE = 10
SIGNAL_WEIGHTING_K = 1.0
MAX_POSITION_WEIGHT = 0.05
MIN_EFFECTIVE_WEIGHT = 0.005
MAX_TURNOVER_DESIGN = 0.25
REBALANCING_DAYS_FIXED = 21  # fixed to isolate the horizon effect (see 09 §5)
BASE_TRANSACTION_COST = 0.0015  # 15 bps, same as notebook 09

# --- Paths -------------------------------------------------------------------
UNIVERSE_PATH = "../data/universe_496.json"
MODEL_CONFIG_PATH = "../data/model_results/final_model/final_model_configurations.json"
TRAIN_FEATURES_PATH = "../data/preprocessed/df_final_rank.parquet"
PRICES_PATH = "../data/raw/sp500_prices_extended.parquet"

def load_frozen_universe(path=UNIVERSE_PATH):
    """Load the frozen ticker universe as a set."""
    with open(path, "r") as f:
        return set(json.load(f))


def load_rf_params(path=MODEL_CONFIG_PATH):
    """Load the tuned Random Forest hyperparameters of the final model."""
    with open(path, "r") as f:
        rf_config = json.load(f)["Random Forest"]
    params = rf_config["hyperparameters"].copy()
    params["random_state"] = rf_config["random_state"]
    params["n_jobs"] = -1
    return params


def load_train_features(universe, path=TRAIN_FEATURES_PATH):
    """Load ranked training features restricted to the frozen universe."""
    df = pd.read_parquet(path)
    df = df[df.index.get_level_values("ticker").isin(universe)]
    return df[FEATURES_RANK]


def load_extended_prices(universe, path=PRICES_PATH):
    """Load extended prices, keeping only frozen-universe tickers (minus exclusions)."""
    prices = pd.read_parquet(path)
    valid_tickers = [
        t for t in prices.columns.levels[1]
        if t in universe and t not in EXCLUDED_OOS_TICKERS
    ]
    return prices.loc[:, (slice(None), valid_tickers)]


def build_oos_features(prices, simple_returns, log_returns):
    """Compute ranked OOS features (winsorized, then cross-sectionally ranked)."""
    log10_amihud = np.log10(
        compute_amihud_illiquidity(prices=prices, simple_returns=simple_returns)
    )

    factor_matrix = (
        pd.concat(
            {
                "momentum_12_1": compute_12_1_momentum(prices).stack(),
                "upside_volatility": compute_upside_volatility(log_returns).stack(),
                "log10_amihud": log10_amihud.stack(),
            },
            axis=1,
        )
        .rename_axis(["date", "ticker"])
        .reset_index()
    )

    df_win = winsorize_cross_sectional(
        factor_matrix, cols=FACTOR_COLS, p_low=0.01, p_high=0.99
    )
    df_rank = rank_cross_sectional(df_win, cols=[f"{c}_win" for c in FACTOR_COLS])

    X_oos = df_rank[["date", "ticker"] + FEATURES_RANK].set_index(["date", "ticker"])
    return X_oos.loc[X_oos.index.get_level_values("date") >= OOS_START]


def compute_forward_targets(adj_close, horizon):
    """Stacked forward returns for a given horizon, as a (date, ticker) frame."""
    return (
        compute_forward_return(adj_close, horizon=horizon)
        .stack()
        .rename(f"forward_return_{horizon}d")
        .rename_axis(["date", "ticker"])
        .to_frame()
        .dropna()
    )

def fit_and_predict(X_train, X_oos, y_full, horizon, window, rf_params):
    """Fit the Random Forest for one (horizon, window) pair and predict OOS."""
    target_col = f"forward_return_{horizon}d"

    # Training sample: window-specific start, common end
    train_dates = X_train.index.get_level_values("date")
    train_mask = (train_dates >= TRAIN_START[window]) & (train_dates <= TRAIN_END)
    X_sub = X_train.loc[train_mask]

    common_idx = X_sub.index.intersection(y_full.index)
    X_fit = X_sub.loc[common_idx]
    y_fit = y_full.loc[common_idx, target_col]

    # OOS sample: no target availability required
    oos_dates = X_oos.index.get_level_values("date")
    X_pred = X_oos.loc[(oos_dates >= OOS_START) & (oos_dates <= OOS_END)]

    model = RandomForestRegressor(**rf_params)
    model.fit(X_fit, y_fit)

    pred_df = X_pred.copy()
    pred_df["prediction_rf"] = model.predict(X_pred)
    pred_df[target_col] = y_full[target_col].reindex(X_pred.index)  # evaluation only
    return pred_df


def run_prediction_grid(X_train, X_oos, adj_close, rf_params):
    """Run the full (horizon x window) grid; returns {(horizon, window): pred_df}."""
    predictions = {}
    for horizon in HORIZONS:
        print(f"\n--- Horizon: {horizon} days ---")
        y_full = compute_forward_targets(adj_close, horizon)
        for window in WINDOWS:
            print(f"  -> Training window: {window}")
            predictions[(horizon, window)] = fit_and_predict(
                X_train, X_oos, y_full, horizon, window, rf_params
            )
    print("\nPredictions generated for all configurations.")
    return predictions

def build_top_decile_weights(predictions, horizon, window):
    """Signal-weighted long-only weights on the top prediction decile."""
    pct_rank = predictions.groupby(level="date").rank(method="first", pct=True)
    decile = np.ceil(pct_rank * 10).clip(upper=10).astype("Int64")

    selected = predictions.loc[decile >= TOP_DECILE]
    selected_rank = selected.groupby(level="date").rank(method="first", pct=True)

    strength = selected_rank ** SIGNAL_WEIGHTING_K
    weights = strength / strength.groupby(level="date").transform("sum")

    out = weights.rename("weight").reset_index()
    out["model"] = MODEL_LABEL
    out["portfolio"] = f"top_10_{horizon}d_{window}"
    return out[["date", "ticker", "model", "portfolio", "weight"]]


def apply_weight_constraints(all_weights):
    """Apply max position weight and min effective weight per (portfolio, date)."""
    constrained = []
    for (model, portfolio, date), group in all_weights.groupby(
        ["model", "portfolio", "date"], sort=False
    ):
        weights = apply_max_position_weight(
            group["weight"].to_numpy(), max_weight=MAX_POSITION_WEIGHT
        )
        weights = apply_min_effective_weight(weights, min_weight=MIN_EFFECTIVE_WEIGHT)

        constrained.append(
            pd.DataFrame(
                {
                    "date": date,
                    "ticker": group["ticker"].to_numpy(),
                    "model": model,
                    "portfolio": portfolio,
                    "weight": weights,
                }
            )
        )
    return pd.concat(constrained, ignore_index=True)

def compute_rebalance_turnover(rebal_dates, rebalance_wide, returns_aligned):
    """One-way turnover at each rebalance, measured against drifted weights."""
    records = []
    for k in range(1, len(rebal_dates) - 1):
        t_prev, t_curr = rebal_dates[k - 1], rebal_dates[k]
        w_prev = rebalance_wide.loc[t_prev]

        window_rets = returns_aligned.loc[
            (returns_aligned.index > t_prev) & (returns_aligned.index <= t_curr)
        ]
        compounded = ((1.0 + window_rets).prod() - 1.0).reindex(w_prev.index).fillna(0.0)

        drifted = w_prev * (1.0 + compounded)
        drifted_norm = drifted / drifted.sum() if drifted.sum() > 0 else w_prev

        w_target = rebalance_wide.loc[t_curr]
        records.append(
            {"date": t_curr, "turnover": 0.5 * (w_target - drifted_norm).abs().sum()}
        )
    return pd.DataFrame(records).set_index("date")


def cost_by_execution_date(turnover_df, trading_dates, cost_rate):
    """Map turnover costs to the trading day following each rebalance date."""
    exec_dates = []
    for d in turnover_df.index:
        pos = trading_dates.get_loc(d)
        exec_dates.append(
            trading_dates[pos + 1] if (pos + 1) < len(trading_dates) else trading_dates[pos]
        )
    costs = turnover_df.assign(
        execution_date=exec_dates,
        transaction_cost=turnover_df["turnover"] * cost_rate,
    )
    return costs.set_index("execution_date")["transaction_cost"]


def compute_gross_returns_with_drift(rebal_dates, rebalance_wide, returns_aligned, trading_dates):
    """Daily gross portfolio returns with weights drifting between rebalances."""
    daily_returns, daily_dates = [], []

    for i, t_rebal in enumerate(rebal_dates):
        pos_rebal = trading_dates.get_loc(t_rebal)
        if pos_rebal + 1 >= len(trading_dates):
            break

        exec_start = trading_dates[pos_rebal + 1]
        if i + 1 < len(rebal_dates):
            window_dates = trading_dates[
                pos_rebal + 1 : trading_dates.get_loc(rebal_dates[i + 1]) + 1
            ]
        else:
            window_dates = trading_dates[trading_dates.get_loc(exec_start):]

        w_current = rebalance_wide.loc[t_rebal].copy()
        for d in window_dates:
            r_d = returns_aligned.loc[d]
            port_ret = (w_current * r_d).sum()
            daily_returns.append(port_ret)
            daily_dates.append(d)
            w_current = w_current * (1.0 + r_d) / (1.0 + port_ret)

    return pd.Series(daily_returns, index=daily_dates, name="gross_return")


def compute_net_returns(rebalance_weights, simple_returns, oos_end, cost_rate):
    """Net daily returns per (model, portfolio), with drift and transaction costs."""
    returns_exec = simple_returns.loc[:oos_end]
    trading_dates = pd.Index(sorted(returns_exec.index.unique()))

    results = []
    for (model_name, portfolio_name), group in rebalance_weights.groupby(["model", "portfolio"]):
        rebal_dates = sorted(group["date"].unique())
        rebalance_wide = (
            group.pivot(index="date", columns="ticker", values="weight")
            .fillna(0.0)
            .reindex(rebal_dates)
            .fillna(0.0)
        )
        returns_aligned = returns_exec.reindex(columns=rebalance_wide.columns).fillna(0.0)

        turnover_df = compute_rebalance_turnover(rebal_dates, rebalance_wide, returns_aligned)
        cost_series = cost_by_execution_date(turnover_df, trading_dates, cost_rate)
        gross_ret = compute_gross_returns_with_drift(
            rebal_dates, rebalance_wide, returns_aligned, trading_dates
        )

        df_res = gross_ret.to_frame()
        df_res["model"] = model_name
        df_res["portfolio"] = portfolio_name
        df_res["transaction_cost"] = df_res.index.map(cost_series).fillna(0.0)
        df_res["net_return_base"] = df_res["gross_return"] - df_res["transaction_cost"]
        df_res["cumulative_net_return_base"] = (1.0 + df_res["net_return_base"]).cumprod() - 1.0
        results.append(df_res.reset_index().rename(columns={"index": "date"}))

    return pd.concat(results, ignore_index=True)

def parse_portfolio_name(portfolio_name):
    """Extract (horizon, window) from a name like 'top_10_21d_expanding'."""
    horizon_str, window = portfolio_name.replace("top_10_", "").split("d_")
    return int(horizon_str), window