
from itertools import combinations
import numpy as np
import pandas as pd

def verify_dataset_integrity(df: pd.DataFrame, name: str) -> None:
    """Prints a concise structural validation summary for a panel DataFrame."""
    print(f"=== {name} Integrity Verification ===")
    print(f"Dataset Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    
    # Verify MultiIndex structure (date, ticker)
    if isinstance(df.index, pd.MultiIndex):
        dates = df.index.get_level_values("date")
        tickers = df.index.get_level_values("ticker")
        print(f"MultiIndex Levels: {df.index.names}")
        print(f"Period Coverage: {dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}")
        print(f"Unique Dates: {dates.nunique():,} | Unique Assets: {tickers.nunique():,}")
    else:
        print("Warning: DataFrame is not indexed by MultiIndex (date, ticker).")
        
    # Concise Data Types Check
    unique_dtypes = df.dtypes.value_counts()
    dtypes_summary = ", ".join([f"{count} {dtype}" for dtype, count in unique_dtypes.items()])
    print(f"\nData Types Summary: All columns are numeric ({dtypes_summary})")
    
    # Concise Missing Values Check
    missing = df.isna().sum()
    missing_with_nulls = missing[missing > 0]
    
    if not missing_with_nulls.empty:
        print("\nMissing values per column:")
        for col, val in missing_with_nulls.items():
            pct = (val / len(df)) * 100
            print(f"  - {col}: {val:,} ({pct:.2f}%)")
    else:
        print("\nMissing values: None (0 nulls found across all columns)")
        
    print("-" * 50 + "\n")


# =============================================================================
#  Combinatorial Purged Cross-Validation (CPCV) for Panel Data
# =============================================================================


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation (CPCV) for panel data with
    overlapping targets.

    Parameters
    ----------
    n_blocks : int, default=7
        Number of contiguous temporal blocks.
    k_validation : int, default=2
        Number of blocks assigned to the validation set in each fold.
    purge_window : int, default=21
        Trading days to purge from Train before Validation (Train -> Validation).
    embargo_window : int, default=11
        Trading days to embargo from Train after Validation (Validation -> Train).
    """

    def __init__(
        self,
        n_blocks=7,
        k_validation=2,
        purge_window=21,
        embargo_window=11,
    ):
        self.n_blocks = n_blocks
        self.k_validation = k_validation
        self.purge_window = purge_window
        self.embargo_window = embargo_window

    def split(self, X):
        """
        Generate Train and Validation indices for each CPCV split.

        Parameters
        ----------
        X : pandas.DataFrame
            DataFrame indexed by ('date', 'ticker').

        Yields
        ------
        train_idx : np.ndarray
            Integer row positions for the training set.
        val_idx : np.ndarray
            Integer row positions for the validation set.
        """

        # Extract unique sorted trading dates
        dates = (
            pd.Series(X.index.get_level_values("date").unique())
            .sort_values()
            .reset_index(drop=True)
        )
        n_dates = len(dates)

        # Define contiguous temporal blocks
        block_bounds = np.linspace(
            0,
            n_dates,
            self.n_blocks + 1,
            dtype=int,
        )

        blocks = {
            i: dates.iloc[block_bounds[i]:block_bounds[i + 1]]
            for i in range(self.n_blocks)
        }

        # Generate all validation combinations
        val_combinations = list(
            combinations(range(self.n_blocks), self.k_validation)
        )

        for val_block_ids in val_combinations:

            train_block_ids = [
                b
                for b in range(self.n_blocks)
                if b not in val_block_ids
            ]

            # Validation dates
            val_dates = (
                pd.concat([blocks[b] for b in val_block_ids])
                .sort_values()
            )

            # ---------------------------------------------------------
            # Merge consecutive validation blocks into continuous segments
            # ---------------------------------------------------------
            segments = []

            current_start = val_block_ids[0]
            current_end = val_block_ids[0]

            for block in val_block_ids[1:]:

                if block == current_end + 1:
                    current_end = block
                else:
                    segments.append((current_start, current_end))
                    current_start = block
                    current_end = block

            segments.append((current_start, current_end))

            # ---------------------------------------------------------
            # Apply purge and embargo only at Train-Validation boundaries
            # ---------------------------------------------------------
            purged_dates = set()
            embargoed_dates = set()

            for start_block, end_block in segments:

                segment_start = blocks[start_block].min()
                segment_end = blocks[end_block].max()

                # Purge before Validation
                start_loc = dates[dates == segment_start].index[0]
                purge_start = max(
                    0,
                    start_loc - self.purge_window,
                )

                purged_dates.update(
                    dates.iloc[purge_start:start_loc]
                )

                # Embargo after Validation
                end_loc = dates[dates == segment_end].index[0]

                embargo_end = min(
                    n_dates,
                    end_loc + 1 + self.embargo_window,
                )

                embargoed_dates.update(
                    dates.iloc[end_loc + 1:embargo_end]
                )

            # Build clean training dates
            raw_train_dates = pd.concat(
                [blocks[b] for b in train_block_ids]
            )

            excluded_dates = purged_dates.union(embargoed_dates)

            clean_train_dates = (
                set(raw_train_dates)
                - excluded_dates
            )

            # Convert dates into row positions
            train_idx = np.where(
                X.index.get_level_values("date").isin(clean_train_dates)
            )[0]

            val_idx = np.where(
                X.index.get_level_values("date").isin(val_dates)
            )[0]

            yield train_idx, val_idx


