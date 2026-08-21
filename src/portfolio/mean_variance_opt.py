import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

MODEL_MAP = {
    "Ridge": {
        "oof_path": "../data/model_results/oof_predictions/oof_preds_ridge_rank.parquet",
        "oos_col": "prediction_ridge_rank",
    },
    "XGBoost": {
        "oof_path": "../data/model_results/oof_predictions/oof_preds_xgb_rank.parquet",
        "oos_col": "prediction_xgb_rank",
    },
    "Random Forest": {
        "oof_path": "../data/model_results/oof_predictions/oof_preds_rf_rank.parquet",
        "oos_col": "prediction_rf_rank",
    },
}

TARGET_COLUMN = "target"
PREDICTION_COLUMN = "pred"


# =============================================================================
# PHASE 1 PIPELINE EXECUTION
# =============================================================================

def build_phase1_calibration_dataset(
    oos_predictions: pd.DataFrame,
) -> dict:
    """Executes Phase 1 of Signal Calibration:

    1. Computes path-averaged OOF predictions (Train/Validation).
    2. Unifies OOF + OOS directly into a single historical time series per model.
    3. Computes cross-sectional percentile ranks (pct) and universe-relative returns (r_rel).

    Parameters
    ----------
    oos_predictions : pd.DataFrame
        Out-of-sample predictions DataFrame indexed by MultiIndex (date, ticker).

    Returns
    -------
    dict
        Dictionary mapping model names to their unified, normalized calibration DataFrame.
    """
    unified_model_datasets = {}

    for model_name, config in MODEL_MAP.items():
        print("=" * 80)
        print(f"PROCESSING PHASE 1 DATASET: {model_name.upper()}")
        print("=" * 80)

        # ---------------------------------------------------------------------
        # 1. Load and path-average OOF predictions
        # ---------------------------------------------------------------------
        oof_raw = pd.read_parquet(config["oof_path"])

        oof_data = (
            oof_raw.groupby(level=["date", "ticker"])
            .agg(
                pred=(PREDICTION_COLUMN, "mean"),
                target=(TARGET_COLUMN, "first"),
            )
            .assign(sample_type="OOF")
        )

        val_end_date = oof_data.index.get_level_values("date").max()

        print(
            f"✓ OOF range: {oof_data.index.get_level_values('date').min().date()} "
            f"→ {val_end_date.date()} ({len(oof_data):,} obs)"
        )

        # ---------------------------------------------------------------------
        # 2. Process OOS Predictions
        # ---------------------------------------------------------------------
        oos_data = pd.DataFrame(
            {
                "pred": oos_predictions[config["oos_col"]],
                "target": oos_predictions["forward_return_21d"],
                "sample_type": "OOS",
            },
            index=oos_predictions.index,
        ).dropna(subset=["target"])

        oos_start_date = oos_data.index.get_level_values("date").min()

        print(
            f"✓ OOS range: {oos_start_date.date()} "
            f"→ {oos_data.index.get_level_values('date').max().date()} ({len(oos_data):,} obs)"
        )

        # ---------------------------------------------------------------------
        # 3. Concatenate Full History (OOF + OOS)
        # ---------------------------------------------------------------------
        full_df = pd.concat([oof_data, oos_data]).sort_index()

        # ---------------------------------------------------------------------
        # 4. Cross-sectional Percentile Rank & Universe Relative Returns
        # ---------------------------------------------------------------------
        # Percentile Rank [0, 1] per date
        full_df["pct"] = full_df.groupby(level="date")["pred"].rank(
            pct=True, method="average"
        )

        # Cross-sectional universe mean return per date
        universe_mean = full_df.groupby(level="date")["target"].transform("mean")

        # Relative return: r_i,t^rel = r_i,t - r_bar_t^universe
        full_df["r_rel"] = full_df["target"] - universe_mean

        unified_model_datasets[model_name] = full_df

        print(f"✓ Phase 1 complete. Total unified observations: {len(full_df):,}\n")

    return unified_model_datasets

# =============================================================================
# PHASE 2 FIT OVER EXPANDING WINDOW (SHAPE x SCALE)
# =============================================================================

def fit_phase2_expanding_calibration(
    calibration_datasets: dict,
    calibration_frequency: str = "MS",
    n_bins: int = 10,
    horizon_days: int = 21,
    scale_window_days: int = 85,
) -> tuple[dict, dict]:
    """Executes Phase 2 of Signal Calibration (Shape x Scale Framework):

    1. Fits an IsotonicRegression over an expanding window for SHAPE.
    2. Calculates short-window spread (85d) vs historical spread for SCALE.
    3. Applies OOS-transition shrinkage to SCALE during start of OOS.
    """
    oos_calibrators = {}
    calibrated_datasets = {}

    for model_name, df in calibration_datasets.items():
        print("=" * 80)
        print(f"RUNNING PHASE 2 (SHAPE x SCALE CALIBRATION): {model_name.upper()}")
        print("=" * 80)

        df = df.copy()

        oof_sample = df[df["sample_type"] == "OOF"][["pct", "r_rel"]].dropna()
        oos_sample = df[df["sample_type"] == "OOS"][["pct", "r_rel"]].dropna()

        oos_dates = oos_sample.index.get_level_values("date").unique().sort_values()

        if calibration_frequency == "MS":
            recalibration_dates = (
                pd.Series(oos_dates)
                .groupby(pd.Series(oos_dates).dt.to_period("M"))
                .first()
                .tolist()
            )
        elif calibration_frequency == "W":
            recalibration_dates = (
                pd.Series(oos_dates)
                .groupby(pd.Series(oos_dates).dt.to_period("W-FRI"))
                .first()
                .tolist()
            )
        elif calibration_frequency == "QS":
            recalibration_dates = (
                pd.Series(oos_dates)
                .groupby(pd.Series(oos_dates).dt.to_period("Q"))
                .first()
                .tolist()
            )
        else:
            raise ValueError("Unsupported frequency. Use 'W', 'MS', or 'QS'.")

        model_calibrators = {}

        for cal_date in recalibration_dates:
            # -----------------------------------------------------------------
            # A. Select Matured Observations (Expanding)
            # -----------------------------------------------------------------
            matured_oos_mask = (
                oos_sample.index.get_level_values("date") + pd.offsets.BDay(horizon_days)
                <= cal_date
            )
            available_oos = oos_sample.loc[matured_oos_mask]
            expanding_data = pd.concat([oof_sample, available_oos])

            # -----------------------------------------------------------------
            # B. SHAPE COMPONENT (Isotonic Regression on Relative Returns)
            # -----------------------------------------------------------------
            expanding_data["bin"] = (
                pd.qcut(
                    expanding_data["pct"],
                    q=n_bins,
                    labels=False,
                    duplicates="drop",
                )
                + 1
            )

            curve = (
                expanding_data.groupby("bin")
                .agg(
                    mean_pct=("pct", "mean"),
                    mean_r_rel=("r_rel", "mean"),
                    obs=("r_rel", "count"),
                )
                .reset_index()
            )

            iso_model = IsotonicRegression(increasing=True, out_of_bounds="clip")
            iso_model.fit(
                curve["mean_pct"],
                curve["mean_r_rel"],
                sample_weight=curve["obs"],
            )

            # Historical Spread Reference (D10 - D1)
            d1_hist = curve.loc[curve["bin"] == 1, "mean_r_rel"].values[0]
            d10_hist = curve.loc[curve["bin"] == n_bins, "mean_r_rel"].values[0]
            spread_historical = max(d10_hist - d1_hist, 1e-6)

            # -----------------------------------------------------------------
            # C. SCALE COMPONENT (Recent Window Spread + Shrinkage)
            # -----------------------------------------------------------------
            # Short rolling window of ~85 business days of matured data
            min_date_scale = cal_date - pd.offsets.BDay(scale_window_days)
            scale_window_data = expanding_data[
                expanding_data.index.get_level_values("date") >= min_date_scale
            ]

            if len(scale_window_data) > 0:
                scale_window_data["bin"] = (
                    pd.qcut(
                        scale_window_data["pct"],
                        q=n_bins,
                        labels=False,
                        duplicates="drop",
                    )
                    + 1
                )
                
                recent_curve = scale_window_data.groupby("bin")["r_rel"].mean()
                d1_rec = recent_curve.get(1, 0.0)
                d10_rec = recent_curve.get(n_bins, 0.0)
                spread_recent = d10_rec - d1_rec

                raw_scale = spread_recent / spread_historical
                raw_scale = max(raw_scale, 0.0)  # No invertir la señal si la ventana corta es ruidosa
            else:
                raw_scale = 1.0

            # Shrinkage calculation based on fraction of real OOS observations in scale window
            oos_dates_in_window = available_oos[
                available_oos.index.get_level_values("date") >= min_date_scale
            ].index.get_level_values("date").nunique()

            total_dates_in_window = scale_window_data.index.get_level_values("date").nunique()
            w_oos = oos_dates_in_window / max(total_dates_in_window, 1)

            scale_adjusted = 1.0 + w_oos * (raw_scale - 1.0)

            # Store calibrator state at this date
            model_calibrators[cal_date] = {
                "iso_model": iso_model,
                "scale_adjusted": scale_adjusted,
            }

        oos_calibrators[model_name] = model_calibrators

        # ---------------------------------------------------------------------
        # D. Combined Alpha Inference: alpha_hat = shape(pct) * scale_adjusted
        # ---------------------------------------------------------------------
        df["alpha_hat"] = np.nan
        recal_series = pd.Series(recalibration_dates, index=recalibration_dates)

        # Baseline inference for OOF
        first_cal = model_calibrators[recalibration_dates[0]]
        df.loc[df["sample_type"] == "OOF", "alpha_hat"] = (
            first_cal["iso_model"].predict(df.loc[df["sample_type"] == "OOF", "pct"])
            * first_cal["scale_adjusted"]
        )

        # OOS inference using active calibrator
        for date_val in oos_dates:
            valid_recals = recal_series[recal_series <= date_val]
            active_date = recalibration_dates[0] if len(valid_recals) == 0 else valid_recals.max()

            active_cal = model_calibrators[active_date]
            mask = df.index.get_level_values("date") == date_val
            
            raw_shape = active_cal["iso_model"].predict(df.loc[mask, "pct"])
            df.loc[mask, "alpha_hat"] = raw_shape * active_cal["scale_adjusted"]

        calibrated_datasets[model_name] = df
        print(f"✓ Phase 2 completed for {model_name}. Calibrators generated: {len(model_calibrators)}\n")

    return oos_calibrators, calibrated_datasets


# =============================================================================
# PHASE 3 INDIVIDUAL MODEL SCALE & VARIANCE ESTIMATION
# =============================================================================

def build_phase3_individual_scales(
    calibrated_datasets: dict,
) -> dict[str, pd.DataFrame]:
    """Executes Phase 3: Computes cross-sectional alpha variance per date."""
    print("=" * 80)
    print("RUNNING PHASE 3: INDIVIDUAL MODEL SCALE ESTIMATION (NO ENSEMBLE)")
    print("=" * 80)

    processed_datasets = {}

    for model_name, df in calibrated_datasets.items():
        df_model = df.copy()

        if "r_rel" in df_model.columns:
            df_model = df_model.rename(columns={"r_rel": "target"})

        # Alpha Variance per date: Var_i(\hat{\alpha}_{m, i, t})
        cross_sectional_var = df_model.groupby(level="date")["alpha_hat"].transform("var")
        df_model["alpha_var"] = cross_sectional_var

        cols_to_keep = ["pct", "alpha_hat", "alpha_var", "target", "sample_type"]
        existing_cols = [c for c in cols_to_keep if c in df_model.columns]
        df_final = df_model[existing_cols].dropna(subset=["alpha_hat", "target"])

        processed_datasets[model_name] = df_final

        oof_count = len(df_final[df_final["sample_type"] == "OOF"])
        oos_count = len(df_final[df_final["sample_type"] == "OOS"])
        
        print(f"✓ Processed {model_name}:")
        print(f"  - Total Observations: {len(df_final):,} (OOF: {oof_count:,} | OOS: {oos_count:,})")
        print(f"  - Alpha Var Range: [{df_final['alpha_var'].min():.6f}, {df_final['alpha_var'].max():.6f}]")

    print("\n✓ Phase 3 completed for all individual models.\n")
    return processed_datasets


# =============================================================================
# Mean-Variance / Maximum Sharpe — Helper Function
# =============================================================================

from scipy.optimize import minimize

def compute_maximum_sharpe_weights(
    expected_returns,
    covariance_matrix,
):
    """
    Compute long-only Maximum Sharpe Ratio weights.

    The Maximum Sharpe problem is reformulated as a convex
    minimum-variance problem:

        minimize    x' Σ x

        subject to  μ' x = 1
                    x >= 0

    The resulting solution is normalized to obtain portfolio weights.
    """

    expected_returns = np.asarray(
        expected_returns,
        dtype=float,
    )

    covariance_matrix = np.asarray(
        covariance_matrix,
        dtype=float,
    )

    n_assets = len(expected_returns)

    # -------------------------------------------------------------------------
    # Initial solution
    # -------------------------------------------------------------------------

    initial_x = np.full(
        n_assets,
        1.0 / expected_returns.sum(),
    )

    # -------------------------------------------------------------------------
    # Objective: portfolio variance
    # -------------------------------------------------------------------------

    def objective(x):

        return (
            x
            @ covariance_matrix
            @ x
        )

    # -------------------------------------------------------------------------
    # Analytical gradient
    # -------------------------------------------------------------------------

    def gradient(x):

        return (
            2.0
            * covariance_matrix
            @ x
        )

    # -------------------------------------------------------------------------
    # Expected return constraint
    # -------------------------------------------------------------------------

    constraint = {
        "type": "eq",
        "fun": lambda x:
            expected_returns @ x - 1.0,
        "jac": lambda x:
            expected_returns,
    }

    # -------------------------------------------------------------------------
    # Long-only constraint
    # -------------------------------------------------------------------------

    bounds = [
        (0.0, None)
        for _ in range(n_assets)
    ]

    # -------------------------------------------------------------------------
    # Optimization
    # -------------------------------------------------------------------------

    result = minimize(
        objective,
        initial_x,
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraint,
        options={
            "maxiter": 500,
            "ftol": 1e-9,
        },
    )

    if not result.success:
        raise RuntimeError(
            f"Maximum Sharpe optimization failed: "
            f"{result.message}"
        )

    # -------------------------------------------------------------------------
    # Convert auxiliary solution into portfolio weights
    # -------------------------------------------------------------------------

    weights = (
        result.x
        / result.x.sum()
    )

    return weights


# =============================================================================
# Isotonic calibration
# =============================================================================

from sklearn.isotonic import IsotonicRegression

def fit_isotonic_calibration(
    calibration_data,
):
    """
    Fit isotonic regression from cross-sectional
    prediction percentile to realized forward return.
    """

    # -------------------------------------------------------------------------
    # Aggregate realized returns by decile
    # -------------------------------------------------------------------------

    decile_data = (
        calibration_data
        .groupby("decile")
        .agg(
            percentile=("percentile", "mean"),
            target=("target", "mean"),
            observations=("target", "count"),
        )
        .reset_index()
    )

    # -------------------------------------------------------------------------
    # Fit isotonic regression
    # -------------------------------------------------------------------------

    isotonic_model = IsotonicRegression(
        increasing=True,
        out_of_bounds="clip",
    )

    isotonic_model.fit(
        decile_data["percentile"],
        decile_data["target"],
        sample_weight=decile_data["observations"],
    )

    return (
        isotonic_model,
        decile_data,
    )