"""Feature set definitions.

This module defines thesis-consistent feature subsets that are robust on
sparse district-week panels (where long rolling windows create many NaNs).

The goal is *not* to change feature computation, but to provide a stable
subset for Track A baselines when the full mechanistic/EWS set is too sparse.
"""

from __future__ import annotations

from typing import Iterable, List, Literal, Sequence


FeatureSetName = Literal["full", "core"]


# NOTE: These are *names* of engineered columns produced by the pipeline.
# The selection helper intersects this list with the DataFrame columns so the
# code is resilient to minor naming/version differences.
CORE_FEATURE_SET_V01: Sequence[str] = (
    # Seasonal
    "feat_week_sin",
    "feat_week_cos",
    "feat_quarter",
    "feat_is_monsoon",

    # Spatial
    "feat_lat_norm",
    "feat_lon_norm",
    "feat_lat_lon_interact",

    # Short-history case dynamics (avoid long-history baselines)
    "feat_cases_lag_1",
    "feat_cases_lag_2",
    "feat_cases_lag_4",
    "feat_cases_ma_4w",
    "feat_cases_growth_rate",
    "feat_cases_var_4w",

    # Short-history climate (avoid long lags that are often missing)
    "feat_temp_lag_1",
    "feat_temp_lag_2",
    "feat_rain_lag_1",
    "feat_rain_lag_2",
    "feat_temp_anomaly",
    "feat_rain_persist_4w",
    "feat_degree_days",
)


def select_feature_columns(
    all_columns: Iterable[str],
    feature_set: FeatureSetName = "full",
) -> List[str]:
    """Select feature columns from an iterable of column names.

    Args:
        all_columns: Column names from a DataFrame.
        feature_set: "full" keeps all `feat_` columns; "core" keeps a
            thesis-consistent subset designed for sparse panels.

    Returns:
        Ordered list of selected feature columns.
    """
    columns = list(all_columns)

    if feature_set == "full":
        return [c for c in columns if c.startswith("feat_")]

    if feature_set != "core":
        raise ValueError(f"Unknown feature_set: {feature_set}")

    present = set(columns)
    selected = [c for c in CORE_FEATURE_SET_V01 if c in present]

    # Fallback: if something went wrong and none of the core columns exist,
    # avoid silently training on an empty matrix.
    if not selected:
        selected = [c for c in columns if c.startswith("feat_")]

    return selected
