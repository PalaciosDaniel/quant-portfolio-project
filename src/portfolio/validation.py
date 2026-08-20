import numpy as np
import pandas as pd


def validate_signal_portfolios(
    weights_df: pd.DataFrame,
    quantile_data: pd.DataFrame,
    quantile_columns: dict,
) -> None:
    """Performs an end-to-end integrity, exposure, position count, and concentration

    validation for signal-weighted portfolios.

    Parameters
    ----------
    weights_df : pd.DataFrame
        DataFrame containing portfolio weights with columns ['date', 'ticker',
        'model', 'portfolio', 'weight'].
    quantile_data : pd.DataFrame
        DataFrame containing cross-sectional quantile assignments per date and
        ticker.
    quantile_columns : dict
        Mapping of model names to their respective quantile column in
        `quantile_data`.
    """
    print("=" * 80)
    print("PORTFOLIO VALIDATION & INTEGRITY REPORT")
    print("=" * 80)

    # =========================================================================
    # 1. Data Integrity (Non-negativity & Duplicates)
    # =========================================================================
    assert (
        weights_df["weight"] >= 0
    ).all(), "Integrity Error: Negative weights detected in long-only portfolio."

    keys = ["date", "ticker", "model", "portfolio"]
    duplicates = weights_df.duplicated(subset=keys).sum()
    assert duplicates == 0, (
        f"Integrity Error: Found {duplicates} duplicated "
        "date-ticker-model-portfolio observations."
    )

    print(
        "✓ Data Integrity: No negative weights or duplicate observations found."
    )

    # =========================================================================
    # 2. Exposure & Signal Differentiation
    # =========================================================================
    exposure = weights_df.groupby(["date", "model", "portfolio"])[
        "weight"
    ].sum()
    assert np.allclose(
        exposure.values, 1.00
    ), "Exposure Error: Total portfolio exposure is not equal to 1.00."

    weight_stats = weights_df.groupby(["date", "model", "portfolio"])[
        "weight"
    ].agg(["min", "max"])
    assert (weight_stats["min"] < weight_stats["max"]).all(), (
        "Signal Error: Signal weighting failed to produce differentiated"
        " weights."
    )

    print(
        "✓ Exposure & Signal: Target exposure met (1.00) and weights are"
        " properly differentiated."
    )

    # =========================================================================
    # 3. Position Count & Quantile Alignment
    # =========================================================================
    portfolio_map = {
        "long_only_top_10_signal_weight": 10,
        "long_only_top_20_signal_weight": 9,
        "long_only_top_30_signal_weight": 8,
    }

    print("\n--- POSITION COUNT BY MODEL ---")
    print()
    for model, q_col in quantile_columns.items():
        quantiles = quantile_data[q_col]
        counts_summary = []

        for p_name, min_q in portfolio_map.items():
            expected = quantiles.groupby(level="date").apply(
                lambda x: (x >= min_q).sum()
            )
            actual = (
                weights_df[
                    (weights_df["model"] == model)
                    & (weights_df["portfolio"] == p_name)
                ]
                .groupby("date")["ticker"]
                .nunique()
            )

            assert actual.equals(expected), (
                f"Selection Mismatch: Inconsistent position counts for {model}"
                f" in {p_name}."
            )
            pct_label = p_name.split("_")[2].upper()
            counts_summary.append(
                f"Top {pct_label}% = {actual.min()}–{actual.max()}"
            )

        print(f"✓ {model}: " + " | ".join(counts_summary))

    # =========================================================================
    # 4. Weight Range & Concentration Ordering
    # =========================================================================
    print("\n--- WEIGHT RANGE & CONCENTRATION ORDERING ---")
    print()
    for p_name in portfolio_map.keys():
        p_weights = weights_df.loc[weights_df["portfolio"] == p_name, "weight"]
        print(
            f"✓ {p_name}: Weight range = [{p_weights.min():.6f},"
            f" {p_weights.max():.6f}]"
        )

    max_w = (
        weights_df.groupby(["date", "model", "portfolio"])["weight"]
        .max()
        .unstack("portfolio")
    )
    assert (
        max_w["long_only_top_10_signal_weight"]
        >= max_w["long_only_top_20_signal_weight"]
    ).all() and (
        max_w["long_only_top_20_signal_weight"]
        >= max_w["long_only_top_30_signal_weight"]
    ).all(), (
        "Concentration Error: Monotonicity of maximum weights violated across"
        " selection levels."
    )

    print(
        "✓ Concentration: Monotonic increase in maximum weight verified (Top"
        " 30% → Top 20% → Top 10%)."
    )
    print()
    print("=" * 80)