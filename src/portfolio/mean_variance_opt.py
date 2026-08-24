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
                "raw_scale": raw_scale,
                "w_oos": w_oos, 
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
# PHASE 4 EXPECTED ALPHA VALIDATION
# =============================================================================

def validate_expected_alphas(
    df: pd.DataFrame, models: list, z_threshold: float = 4.0
) -> None:
    """Validates summary statistics and identifies cross-sectional outliers for OOS expected

    alphas across specified models.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing expected alphas (must include 'date' in its index).
    models : list
        List of model names (e.g., PREDICTION_COLUMNS).
    z_threshold : float, optional
        Z-score threshold within each date to flag extreme outliers (default is 4.0).
    """
    print("=" * 80)
    print("OOS EXPECTED ALPHA SUMMARY STATISTICS & CALIBRATION ACCURACY")
    print("=" * 80)

    for model in models:
        alpha_column = f"expected_alpha_{model.lower().replace(' ', '_')}"

        if alpha_column not in df.columns:
            print(
                f"[WARNING] Column '{alpha_column}' not found in DataFrame. Skipping {model}."
            )
            continue

        alpha = df[alpha_column]
        valid_count = alpha.notna().sum()
        missing_count = alpha.isna().sum()

        mean_val = alpha.mean()
        std_val = alpha.std()
        min_val = alpha.min()
        max_val = alpha.max()
        p95_val = alpha.quantile(0.95)
        p99_val = alpha.quantile(0.99)

        # Cross-sectional Z-score calculation per date
        z_scores = df.groupby(level="date")[alpha_column].transform(
            lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) > 0 else 0
        )

        n_outliers = (z_scores.abs() > z_threshold).sum()
        pct_outliers = (n_outliers / valid_count) if valid_count > 0 else 0.0

        # Console report output
        print(
            f"{model}: "
            f"Missing = {missing_count:,} | "
            f"Mean = {mean_val:.6f} | "
            f"Std = {std_val:.6f} | "
            f"Min = {min_val:.6f} | "
            f"Max = {max_val:.6f} | "
            f"P95 = {p95_val:.6f} | "
            f"P99 = {p99_val:.6f}"
        )

        print(
            f"   Outliers (|z| > {z_threshold:.0f} within date) = {n_outliers:,} "
            f"({pct_outliers:.2%})"
        )

    print("=" * 80)


def validate_alpha_decile_monotonicity(
    df: pd.DataFrame, models: list, return_col: str = "forward_return_21d"
) -> None:
    """Validates cross-sectional alpha rankings by evaluating pooled decile returns,

    D10-D1 spreads, overall monotonicity, and monthly monotonicity rates across models.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing expected alphas and realized forward returns
        (must include 'date' in its index).
    models : list
        List of model names (e.g., PREDICTION_COLUMNS).
    return_col : str, optional
        Column name for forward realized returns (default is 'forward_return_21d').
    """
    print("=" * 80)
    print("OOS ALPHA DECILE MONOTONICITY & RANKING EFFICIENCY")
    print("=" * 80)

    for model in models:
        alpha_column = f"expected_alpha_{model.lower().replace(' ', '_')}"

        if (
            alpha_column not in df.columns
            or return_col not in df.columns
        ):
            print(
                f"[WARNING] Required columns ('{alpha_column}' or '{return_col}') not found. Skipping {model}."
            )
            continue

        data = df[[alpha_column, return_col]].copy()

        # Assign alpha deciles cross-sectionally per date
        data["alpha_decile"] = (
            data[alpha_column]
            .groupby(level="date")
            .rank(method="first", pct=True)
            .mul(10)
            .apply(np.ceil)
            .clip(upper=10)
            .astype(int)
        )

        # Mean realized return by decile (pooled)
        decile_returns = data.groupby("alpha_decile")[
            return_col
        ].mean()

        d1_return = decile_returns.loc[1]
        d10_return = decile_returns.loc[10]
        spread = d10_return - d1_return

        # Check global (pooled) monotonicity across deciles
        monotonic = np.all(np.diff(decile_returns.to_numpy()) >= 0)

        print(f"\nModel: {model}")
        print(f"✓ D1 mean return                  = {d1_return:.6%}")
        print(f"✓ D10 mean return                 = {d10_return:.6%}")
        print(f"✓ D10 - D1 spread                 = {spread:.6%}")
        print(
            f"✓ Monotonic relationship (pooled) = {monotonic}"
        )

        # Monthly monotonicity rate
        monthly_decile_returns = (
            data.groupby(
                [
                    data.index.get_level_values("date").to_period(
                        "M"
                    ),
                    "alpha_decile",
                ]
            )[return_col]
            .mean()
            .unstack("alpha_decile")
            .reindex(columns=range(1, 11))
        )

        monthly_monotonic = monthly_decile_returns.apply(
            lambda row: np.all(
                np.diff(row.dropna().to_numpy()) >= 0
            )
            if row.notna().sum() >= 2
            else np.nan,
            axis=1,
        )

        n_months = monthly_monotonic.notna().sum()
        mono_rate = (
            monthly_monotonic.mean() if n_months > 0 else 0.0
        )

        print(
            f"✓ Monthly monotonicity rate       = {mono_rate:.1%} "
            f"({n_months} months evaluated)"
        )

    print("\n" + "=" * 80)


def validate_scale_adjusted_evolution(
    calibrators: dict, models: list
) -> None:
    """Validates scale_adjusted temporal evolution to verify shrink-to-neutral behavior

    over time across specified models.

    Parameters:
    -----------
    calibrators : dict
        Nested dictionary containing OOS calibrators per model and calibration date
        (e.g., oos_calibrators[model][cal_date]['scale_adjusted']).
    models : list
        List of model names (e.g., PREDICTION_COLUMNS).
    """
    print("=" * 80)
    print("SCALE_ADJUSTED EVOLUTION — SHRINK-TO-NEUTRAL CHECK")
    print("=" * 80)

    for model in models:
        if model not in calibrators or not calibrators[model]:
            print(
                f"[WARNING] Calibrator data for '{model}' not found. Skipping."
            )
            continue

        calib_dates = sorted(calibrators[model].keys())
        scale_series = pd.Series(
            {
                cal_date: calibrators[model][cal_date][
                    "scale_adjusted"
                ]
                for cal_date in calib_dates
            }
        )

        first_scale = scale_series.iloc[0]
        last_scale = scale_series.iloc[-1]

        early_dev = (scale_series.iloc[:4] - 1.0).abs().mean()
        late_dev = (scale_series.iloc[4:] - 1.0).abs().mean()
        shrink_as_expected = early_dev <= late_dev

        print(f"\nModel: {model}")
        print(
            f"✓ First calibration scale_adjusted              = {first_scale:.4f} (expected: close to 1.0)"
        )
        print(
            f"✓ Last calibration scale_adjusted               = {last_scale:.4f}"
        )
        print(
            f"✓ Mean |scale_adjusted - 1.0| (first 4 months)  = {early_dev:.4f}"
        )
        print(
            f"✓ Mean |scale_adjusted - 1.0| (after 4 months)   = {late_dev:.4f}"
        )
        print(
            f"✓ Shrink behaving as expected (early <= late)   = {shrink_as_expected}"
        )

    print("\n" + "=" * 80)


def validate_equal_weight_fallback_rate(
    df: pd.DataFrame, models: list
) -> None:
    """Calculates the frequency at which all expected alphas are non-positive on a given date,

    triggering an equal-weight portfolio fallback.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing expected alphas (must include 'date' in its index).
    models : list
        List of model names (e.g., PREDICTION_COLUMNS).
    """
    print("=" * 80)
    print("FRACTION OF DATES TRIGGERING EQUAL-WEIGHT FALLBACK")
    print("=" * 80)

    for model in models:
        alpha_column = f"expected_alpha_{model.lower().replace(' ', '_')}"

        if alpha_column not in df.columns:
            print(
                f"[WARNING] Column '{alpha_column}' not found in DataFrame. Skipping {model}."
            )
            continue

        all_nonpositive_by_date = df.groupby(level="date")[
            alpha_column
        ].apply(
            lambda x: (x.dropna() <= 0).all()
            if x.notna().any()
            else np.nan
        )

        n_fallback_dates = int(all_nonpositive_by_date.sum())
        n_total_dates = all_nonpositive_by_date.notna().sum()
        fallback_rate = (
            (n_fallback_dates / n_total_dates)
            if n_total_dates > 0
            else 0.0
        )

        print(
            f"{model}: Fallback triggered on {n_fallback_dates:,} / {n_total_dates:,} "
            f"dates ({fallback_rate:.2%})"
        )

    print("=" * 80)


def validate_alpha_cross_sectional_variance(
    df: pd.DataFrame, models: list, near_zero_tol: float = 1e-12
) -> None:
    """Validates the daily cross-sectional variance of expected alphas, which serves

    as a scaling input for Mean-Variance Optimization (MVO).

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing expected alpha variance metrics (must include 'date' in its index).
    models : list
        List of model names (e.g., PREDICTION_COLUMNS).
    near_zero_tol : float, optional
        Tolerance threshold below which daily variance is considered near-zero (default is 1e-12).
    """
    print("=" * 80)
    print("ALPHA CROSS-SECTIONAL VARIANCE (MVO SCALE INPUT) SANITY CHECK")
    print("=" * 80)

    for model in models:
        var_column = f"alpha_var_{model.lower().replace(' ', '_')}"

        if var_column not in df.columns:
            print(
                f"[WARNING] Column '{var_column}' not found in DataFrame. Skipping {model}."
            )
            continue

        daily_var = df.groupby(level="date")[var_column].first()

        total_dates = len(daily_var)
        n_near_zero = (daily_var < near_zero_tol).sum()
        pct_near_zero = (
            (n_near_zero / total_dates) if total_dates > 0 else 0.0
        )

        mean_var = daily_var.mean()
        min_var = daily_var.min()
        p5_var = daily_var.quantile(0.05)

        print(
            f"{model}: "
            f"Mean = {mean_var:.8f} | "
            f"Min = {min_var:.8f} | "
            f"P5 = {p5_var:.8f} | "
            f"Near-zero dates (< {near_zero_tol:.0e}) = {n_near_zero:,} "
            f"({pct_near_zero:.2%})"
        )

    print("=" * 80)


def _safe_spearman(g: pd.DataFrame, col1: str, col2: str) -> float:
    """Calculates Spearman rank correlation safely for a group, returning NaN

    if data has fewer than 2 valid observations or zero variance.
    """
    s1, s2 = g[col1].dropna(), g[col2].dropna()
    common_idx = s1.index.intersection(s2.index)

    if (
        len(common_idx) < 2
        or s1.loc[common_idx].nunique() < 2
        or s2.loc[common_idx].nunique() < 2
    ):
        return np.nan

    return s1.loc[common_idx].corr(s2.loc[common_idx], method="spearman")


def validate_cross_model_rank_agreement(
    df: pd.DataFrame, models: list, low_corr_threshold: float = 0.3
) -> None:
    """Calculates daily cross-sectional Spearman rank correlation between all pairs

    of models to assess rank agreement over time.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing expected alphas (must include 'date' in its index).
    models : list
        List of model names (e.g., list(PREDICTION_COLUMNS.keys()) or PREDICTION_COLUMNS).
    low_corr_threshold : float, optional
        Correlation threshold below which rank agreement is considered low (default is 0.3).
    """
    print("=" * 80)
    print("CROSS-MODEL RANK AGREEMENT (SPEARMAN, BY DATE)")
    print("=" * 80)

    # Handle dictionary keys or list of strings for models input
    model_names = (
        list(models.keys()) if isinstance(models, dict) else list(models)
    )

    alpha_cols = {
        model: f"expected_alpha_{model.lower().replace(' ', '_')}"
        for model in model_names
    }

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m1, m2 = model_names[i], model_names[j]
            col1, col2 = alpha_cols[m1], alpha_cols[m2]

            if col1 not in df.columns or col2 not in df.columns:
                print(
                    f"[WARNING] Columns '{col1}' or '{col2}' not found in DataFrame. Skipping pair ({m1} vs {m2})."
                )
                continue

            daily_corr = df.groupby(level="date")[[col1, col2]].apply(
                lambda g: _safe_spearman(g, col1, col2)
            )

            n_skipped = daily_corr.isna().sum()
            total_dates = len(daily_corr)
            valid_corr = daily_corr.dropna()

            if len(valid_corr) == 0:
                print(
                    f"{m1} vs {m2}: No valid dates with sufficient variation for Spearman correlation."
                )
                continue

            pct_skipped = (
                (n_skipped / total_dates) if total_dates > 0 else 0.0
            )
            pct_low_corr = (valid_corr < low_corr_threshold).mean()

            print(
                f"{m1} vs {m2}: "
                f"Mean rank corr = {valid_corr.mean():.4f} | "
                f"Min = {valid_corr.min():.4f} | "
                f"% dates corr < {low_corr_threshold:.1f} = {pct_low_corr:.2%} | "
                f"Dates skipped = {n_skipped:,} ({pct_skipped:.2%})"
            )

    print("=" * 80)


# =============================================================================
# Mean-Variance / Maximum Sharpe — Helper Function
# =============================================================================

from scipy.optimize import minimize
from src.portfolio.constraints import apply_max_position_weight
from src.portfolio.constraints import apply_min_effective_weight


def compute_maximum_sharpe_weights(
    expected_returns,
    covariance_matrix,
    max_weight=0.05,
    min_weight=0.005,
):
    """Compute long-only Maximum Sharpe Ratio weights via quadratic reformulation.

        The maximum position cap (`max_weight`) is enforced exactly, as a linear
    inequality constraint inside the optimization problem itself. The minimum
    effective weight (`min_weight`) is enforced as a post-processing heuristic:
    imposing "each position is either 0 or >= min_weight" exactly would turn
    this into a non-convex, semicontinuous (mixed-integer) problem, which is
    far more expensive to solve. Thresholding and renormalizing after the
    fact is the standard industry approximation for this type of constraint.

    Parameters
    ----------
    expected_returns : array-like
        Expected alpha or return vector per asset.
    covariance_matrix : array-like
        Asset covariance matrix (N x N).
    max_weight : float, optional
        Maximum allowed position weight (e.g. 0.05 for 5%), enforced exactly
        inside the optimizer, by default 0.05.
    min_weight : float, optional
        Minimum effective position weight below which positions are zeroed out,
        applied as post-hoc heuristic, by default 0.005 (0.5%).

    Returns
    -------
    weights : np.ndarray
        Constrained long-only portfolio weights summing to 1.0.
    used_fallback : bool
        True if the equal-weight fallback was used (non-positive/NaN alpha,
        or optimizer non-convergence) instead of a genuine Sharpe optimum.
    """
    expected_returns = np.asarray(expected_returns, dtype=float)
    covariance_matrix = np.asarray(covariance_matrix, dtype=float)
    n_assets = len(expected_returns)

    # 1. Fallback: If all expected returns are non-positive or NaN
    if np.all(expected_returns <= 0) or np.isnan(expected_returns).any():
        weights = np.full(n_assets, 1.0 / n_assets)
        if max_weight is not None:
            weights = apply_max_position_weight(
                weights, max_weight=max_weight
            )
        return weights, True

    # 2. Safe Initial Solution over positive returns
    pos_mask = expected_returns > 0
    initial_x = np.zeros(n_assets)
    if np.any(pos_mask):
        initial_x[pos_mask] = 1.0 / expected_returns[pos_mask].sum()
    else:
        initial_x = np.full(n_assets, 1.0 / n_assets)

    # 3. Core Convex Optimization
    def objective(x):
        return x @ covariance_matrix @ x

    def gradient(x):
        return 2.0 * (covariance_matrix @ x)

    constraints = [
        {
            "type": "eq",
            "fun": lambda x: expected_returns @ x - 1.0,
            "jac": lambda x: expected_returns,
        }
    ]

    if max_weight is not None:
        for i in range(n_assets):
            constraints.append({
                "type": "ineq",
                "fun": lambda x, i=i: max_weight * np.sum(x) - x[i],
                "jac": lambda x, i=i: max_weight * np.ones(n_assets)
                - np.eye(n_assets)[i],
            })

    bounds = [(0.0, None) for _ in range(n_assets)]

    weights = None
    try:
        result = minimize(
            objective,
            initial_x,
            jac=gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-9},
        )

        if result.success and result.x.sum() > 1e-8:
            weights = result.x / result.x.sum()
    except Exception:
        pass

    used_fallback = False
    if weights is None:
        weights = np.full(n_assets, 1.0 / n_assets)
        if max_weight is not None:
            weights = apply_max_position_weight(weights, max_weight=max_weight)
        used_fallback = True

    # 4. Post-processing: Apply minimum effective weight threshold
    if min_weight is not None and min_weight > 0:
        try:
            weights = apply_min_effective_weight(
                weights, min_weight=min_weight
            )
        except ValueError:
            pass

    return weights, used_fallback


# =============================================================================
# MAXIMUM SHARPE PORTFOLIO — COMPREHENSIVE VALIDATION & DIAGNOSTICS
# =============================================================================

def build_max_sharpe_diagnostics_df(
    weights_df: pd.DataFrame,
    covariances_dict: dict,
    expected_alpha_df: pd.DataFrame,
    fallback_flags_dict: dict,
    alpha_columns_map: dict,
    max_position_weight: float = 0.10,
    bound_tolerance: float = 1e-6,
) -> pd.DataFrame:
    """Generates the base validation DataFrame with ex-ante metrics and constraint checks."""
    validation_records = []
    grouped = weights_df.groupby(["date", "model", "portfolio"])

    for (date, model, portfolio), group in grouped:
        tickers = group["ticker"].tolist()
        weights = group["weight"].to_numpy()
        n_assets = len(weights)

        cov_key = (date, model, portfolio)
        covariance_matrix = np.asarray(covariances_dict[cov_key])

        alpha_col = alpha_columns_map[model]
        expected_alpha = (
            expected_alpha_df.loc[(date, slice(None)), alpha_col]
            .droplevel("date")
            .reindex(tickers)
            .to_numpy()
        )

        weight_sum_err = abs(weights.sum() - 1.0)
        min_active_w = (
            np.min(weights[weights > bound_tolerance])
            if np.any(weights > bound_tolerance)
            else 0.0
        )
        max_w = np.max(weights)
        non_zero_positions = np.sum(weights > 1e-4)

        max_weight_violation = max(
            0.0, max_w - max_position_weight - bound_tolerance
        )
        max_weight_infeasible = (n_assets * max_position_weight) < (
            1.0 - 1e-9
        )

        min_w = np.min(weights)
        negative_weight_violation = max(0.0, -min_w - bound_tolerance)

        used_fallback = fallback_flags_dict.get(cov_key, np.nan)

        opt_ret = weights @ expected_alpha
        opt_vol = np.sqrt(weights @ covariance_matrix @ weights)
        opt_sharpe = opt_ret / opt_vol if opt_vol > 0 else np.nan

        ew_weights = np.full(n_assets, 1.0 / n_assets)
        ew_ret = ew_weights @ expected_alpha
        ew_vol = np.sqrt(ew_weights @ covariance_matrix @ ew_weights)
        ew_sharpe = ew_ret / ew_vol if ew_vol > 0 else np.nan

        sharpe_delta = opt_sharpe - ew_sharpe

        hhi = np.sum(weights**2)
        n_eff = 1.0 / hhi if hhi > 0 else np.nan
        cond_number = np.linalg.cond(covariance_matrix)

        validation_records.append(
            {
                "date": date,
                "model": model,
                "portfolio": portfolio,
                "n_selected": n_assets,
                "n_active": non_zero_positions,
                "weight_sum_err": weight_sum_err,
                "min_active_weight": min_active_w,
                "max_weight": max_w,
                "max_weight_violation": max_weight_violation,
                "max_weight_infeasible": max_weight_infeasible,
                "negative_weight_violation": negative_weight_violation,
                "used_fallback": used_fallback,
                "hhi": hhi,
                "n_eff": n_eff,
                "cond_number": cond_number,
                "opt_ret": opt_ret,
                "opt_vol": opt_vol,
                "opt_sharpe": opt_sharpe,
                "sharpe_delta": sharpe_delta,
            }
        )

    val_df = pd.DataFrame(validation_records)
    val_df["portfolio_short"] = (
        val_df["portfolio"]
        .str.replace("long_only_", "", regex=False)
        .str.replace("_maximum_sharpe", "", regex=False)
    )

    return val_df


def report_sanity_checks(val_df: pd.DataFrame) -> None:
    """Prints Summary Report 1: Global Constraints & Sanity Checks."""
    print("=" * 80)
    print("1. SANITY CHECKS & CONSTRAINT COMPLIANCE")
    print("=" * 80)
    print(f"Total Portfolios Evaluated      : {len(val_df):,}")
    print(
        f"Max Weight Sum Error            : {val_df['weight_sum_err'].max():.2e}"
    )
    print(
        f"Min Active Weight Observed      : {val_df['min_active_weight'].min():.6f}"
    )
    print(
        f"Max Weight Observed             : {val_df['max_weight'].max():.6f}"
    )
    print(
        f"Max Weight Violations (> tol)   : {(val_df['max_weight_violation'] > 0).sum():,} "
        f"({(val_df['max_weight_violation'] > 0).mean():.2%})"
    )
    print(
        f"Groups with Infeasible Max Cap  : {val_df['max_weight_infeasible'].sum():,} "
        f"({val_df['max_weight_infeasible'].mean():.2%})"
    )
    print(
        f"Negative Weight Violations      : {(val_df['negative_weight_violation'] > 0).sum():,}"
    )
    print(
        f"Equal-Weight Fallback Rate      : {val_df['used_fallback'].mean():.2%}"
    )
    print(
        f"Sharpe Improvement Rate         : {(val_df['sharpe_delta'] >= -1e-6).mean() * 100:.2f}%"
    )
    print("=" * 80)


def report_concentration_and_performance(val_df: pd.DataFrame) -> None:
    """Prints Summary Report 2: Metrics by Portfolio & Model."""
    summary_by_portfolio = val_df.groupby(["model", "portfolio_short"]).agg(
        Avg_Assets=("n_selected", "mean"),
        Avg_Active=("n_active", "mean"),
        Max_Weight_Avg=("max_weight", "mean"),
        Max_Weight_P95=("max_weight", lambda x: x.quantile(0.95)),
        N_Eff_Avg=("n_eff", "mean"),
        Cond_Number_P95=("cond_number", lambda x: x.quantile(0.95)),
        Fallback_Rate=("used_fallback", "mean"),
        ExAnte_Sharpe_Opt=("opt_sharpe", "mean"),
        Sharpe_Gain=("sharpe_delta", "mean"),
    )

    print("=" * 80)
    print("2. CONCENTRATION & EX-ANTE PERFORMANCE SUMMARY")
    print("=" * 80)
    print(summary_by_portfolio.to_string(float_format=lambda x: f"{x:.4f}"))
    print("=" * 80)


def report_raw_turnover(weights_df: pd.DataFrame) -> None:
    """Prints Summary Report 3: Raw Turnover between consecutive rebalance dates."""
    print("=" * 80)
    print("3. RAW TURNOVER BETWEEN CONSECUTIVE REBALANCE DATES (DIAGNOSTIC ONLY)")
    print("=" * 80)
    turnover_records = []
    for (model, portfolio), group in weights_df.groupby(
        ["model", "portfolio"]
    ):
        pivoted = group.pivot_table(
            index="date", columns="ticker", values="weight", fill_value=0.0
        ).sort_index()
        raw_turnover = pivoted.diff().abs().sum(axis=1).iloc[1:]
        turnover_records.append(
            {
                "model": model,
                "portfolio": portfolio,
                "avg_turnover": raw_turnover.mean(),
                "max_turnover": raw_turnover.max(),
            }
        )

    turnover_df = pd.DataFrame(turnover_records)
    print(turnover_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 80)


def report_cross_model_agreement(
    weights_df: pd.DataFrame, low_corr_threshold: float = 0.3
) -> None:
    """Prints Summary Report 4: Cross-Model Rank Agreement on Final Weights using Safe Spearman."""
    print("=" * 80)
    print("4. CROSS-MODEL RANK AGREEMENT ON FINAL WEIGHTS (SPEARMAN, BY DATE)")
    print("=" * 80)
    model_names = list(weights_df["model"].unique())

    for portfolio in weights_df["portfolio"].unique():
        port_data = weights_df[weights_df["portfolio"] == portfolio]
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                m1, m2 = model_names[i], model_names[j]

                w1 = port_data[port_data["model"] == m1].set_index(
                    ["date", "ticker"]
                )["weight"]
                w2 = port_data[port_data["model"] == m2].set_index(
                    ["date", "ticker"]
                )["weight"]

                merged = pd.concat([w1, w2], axis=1, keys=[m1, m2]).dropna()
                if merged.empty:
                    continue

                daily_corr = merged.groupby(level="date").apply(
                    lambda g: _safe_spearman(g, m1, m2)
                )

                valid_corr = daily_corr.dropna()
                n_skipped = daily_corr.isna().sum()
                total_dates = len(daily_corr)

                if len(valid_corr) == 0:
                    print(
                        f"[{portfolio}] {m1} vs {m2}: No valid dates for correlation."
                    )
                    continue

                pct_low_corr = (valid_corr < low_corr_threshold).mean()
                pct_skipped = (
                    (n_skipped / total_dates) if total_dates > 0 else 0.0
                )

                print(
                    f"[{portfolio}] {m1} vs {m2}: "
                    f"Mean rank corr = {valid_corr.mean():.4f} | "
                    f"% dates corr < {low_corr_threshold:.1f} = {pct_low_corr:.2%} | "
                    f"Dates skipped = {n_skipped:,} ({pct_skipped:.2%})"
                )

    print("=" * 80)


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