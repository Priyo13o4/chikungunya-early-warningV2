#!/usr/bin/env python3
"""Experiment 11: Sparsity / Missingness Report

Generates a quantitative report explaining why CV folds become unstable due to
feature/label sparsity, and saves tables under `results/analysis/sparsity_report/`.

This script is designed to be run from the v6 snapshot, but it reads the
repo-level engineered features parquet (shared pipeline artifact).

Usage:
  python versions/Priyo-Version-Hist/v6/experiments/11_sparsity_report.py
  python versions/Priyo-Version-Hist/v6/experiments/11_sparsity_report.py --config config/config_default.yaml

Outputs:
  - missingness_overall.csv
  - missingness_labeled.csv
  - feature_strict_impact.csv
  - district_year_week_counts.csv
    - label_availability_by_year.csv
    - label_availability_by_district.csv
    - label_missing_reason_counts.csv
  - SPARSITY_REPORT.md
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# Add v6 snapshot root to sys.path
V6_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(V6_ROOT))

from src.config import get_project_root, get_repo_root, load_config
from src.features.feature_sets import CORE_FEATURE_SET_V01, select_feature_columns


@dataclass
class DatasetStats:
    n_rows: int
    n_states: int
    n_districts: int
    year_min: int | float
    year_max: int | float
    n_labeled: int
    n_pos: int


def _safe_nunique(df: pd.DataFrame, col: str) -> int:
    return int(df[col].nunique()) if col in df.columns else 0


def compute_dataset_stats(df: pd.DataFrame) -> DatasetStats:
    labeled = df[df["label_outbreak"].notna()] if "label_outbreak" in df.columns else df.iloc[0:0]
    n_pos = int((labeled["label_outbreak"] == 1).sum()) if "label_outbreak" in df.columns else 0

    return DatasetStats(
        n_rows=int(len(df)),
        n_states=_safe_nunique(df, "state"),
        n_districts=_safe_nunique(df, "district"),
        year_min=df["year"].min() if "year" in df.columns else np.nan,
        year_max=df["year"].max() if "year" in df.columns else np.nan,
        n_labeled=int(labeled.shape[0]),
        n_pos=n_pos,
    )


def missingness_table(df: pd.DataFrame, cols: List[str], prefix: str) -> pd.DataFrame:
    miss_count = df[cols].isna().sum().rename(f"{prefix}_missing_count")
    miss_pct = (df[cols].isna().mean() * 100).rename(f"{prefix}_missing_pct")
    present = df[cols].notna().sum().rename(f"{prefix}_present_count")
    return (
        pd.concat([present, miss_count, miss_pct], axis=1)
        .reset_index()
        .rename(columns={"index": "feature"})
        .sort_values(f"{prefix}_missing_pct", ascending=False)
        .reset_index(drop=True)
    )


def strict_impact_table(label_df: pd.DataFrame, feat_cols: List[str]) -> pd.DataFrame:
    """Quantify which single features most reduce strict-complete-case rows."""
    strict_all = label_df.dropna(subset=feat_cols)
    n_strict_all = int(len(strict_all))

    rows = []
    for feature in feat_cols:
        subset = [c for c in feat_cols if c != feature]
        n_without = int(len(label_df.dropna(subset=subset))) if subset else int(len(label_df))
        rows.append(
            {
                "feature": feature,
                "labeled_non_na": int(label_df[feature].notna().sum()),
                "labeled_missing": int(label_df[feature].isna().sum()),
                "strict_all_rows": n_strict_all,
                "strict_rows_without_feature": n_without,
                "delta_if_drop_feature": int(n_without - n_strict_all),
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(["delta_if_drop_feature", "labeled_missing"], ascending=False).reset_index(drop=True)
    return out


def district_year_week_counts(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["state", "district", "year"]
    if not all(k in df.columns for k in keys):
        return pd.DataFrame()

    g = df.groupby(keys, dropna=False)
    out = g.agg(
        n_rows=("week", "size"),
        n_weeks_unique=("week", "nunique"),
        week_min=("week", "min"),
        week_max=("week", "max"),
    ).reset_index()

    # A crude coverage proxy: unique weeks divided by observed span.
    span = (out["week_max"] - out["week_min"] + 1).replace(0, np.nan)
    out["week_coverage_proxy"] = (out["n_weeks_unique"] / span).round(4)

    return out.sort_values(["n_rows"], ascending=False).reset_index(drop=True)


def label_availability_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize label availability per year."""
    if "year" not in df.columns or "label_outbreak" not in df.columns:
        return pd.DataFrame()

    g = df.groupby("year", dropna=False)
    out = g.agg(
        n_rows=("label_outbreak", "size"),
        n_labeled=("label_outbreak", lambda s: int(s.notna().sum())),
        n_pos=("label_outbreak", lambda s: int((s == 1).sum())),
    ).reset_index()
    out["labeled_pct"] = (100 * out["n_labeled"] / out["n_rows"]).round(2)
    out["pos_rate_pct"] = (100 * out["n_pos"] / out["n_labeled"]).replace([np.inf, -np.inf], np.nan).round(2)
    return out.sort_values("year").reset_index(drop=True)


def label_availability_by_district(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize label availability per district (state+district)."""
    keys = ["state", "district"]
    if not all(k in df.columns for k in keys) or "label_outbreak" not in df.columns:
        return pd.DataFrame()

    g = df.groupby(keys, dropna=False)
    out = g.agg(
        n_rows=("label_outbreak", "size"),
        n_labeled=("label_outbreak", lambda s: int(s.notna().sum())),
        n_pos=("label_outbreak", lambda s: int((s == 1).sum())),
    ).reset_index()
    out["labeled_pct"] = (100 * out["n_labeled"] / out["n_rows"]).round(2)
    out["pos_rate_pct"] = (100 * out["n_pos"] / out["n_labeled"]).replace([np.inf, -np.inf], np.nan).round(2)
    return out.sort_values(["n_labeled", "n_rows"], ascending=False).reset_index(drop=True)


def label_missing_reason_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Infer why labels are missing.

    In v6 dynamic labeling, `label_outbreak` is NaN when the per-row threshold is NaN.
    We infer major causes:
    - early_in_series: within first 10 observations for a district
    - insufficient_valid_history: 10+ observations but threshold is NaN (likely too many NaNs in incidence history)
    - other: label missing but threshold present (unexpected)
    """
    keys = ["state", "district"]
    needed = set(keys + ["label_outbreak"])
    if not needed.issubset(df.columns):
        return pd.DataFrame()

    has_threshold = "label_threshold" in df.columns

    work = df.copy()
    # Index within each district time series (0-based)
    work["_idx_in_series"] = work.groupby(keys, sort=False).cumcount()

    missing_label = work["label_outbreak"].isna()
    if has_threshold:
        threshold_missing = work["label_threshold"].isna()
    else:
        # If threshold isn't present, we can only attribute to early vs unknown
        threshold_missing = pd.Series(True, index=work.index)

    early = missing_label & threshold_missing & (work["_idx_in_series"] < 10)
    insufficient_hist = missing_label & threshold_missing & (work["_idx_in_series"] >= 10)
    other = missing_label & (~threshold_missing)
    labeled = ~missing_label

    rows = [
        {"reason": "labeled", "n_rows": int(labeled.sum())},
        {"reason": "early_in_series(<10)", "n_rows": int(early.sum())},
        {"reason": "insufficient_valid_history(>=10)", "n_rows": int(insufficient_hist.sum())},
        {"reason": "other_unexpected", "n_rows": int(other.sum())},
    ]
    out = pd.DataFrame(rows)
    out["pct_of_all_rows"] = (100 * out["n_rows"] / len(work)).round(2)
    out["pct_of_missing_labels"] = (
        100 * out["n_rows"] / max(int(missing_label.sum()), 1)
    ).round(2)
    return out


def write_markdown_summary(
    out_path: Path,
    stats: DatasetStats,
    feat_cols: List[str],
    miss_overall: pd.DataFrame,
    miss_labeled: pd.DataFrame,
    impact: pd.DataFrame,
    core_feat_cols: List[str],
    core_strict_rows: int,
    strict_all_rows: int,
) -> None:
    pos_rate = (100 * stats.n_pos / stats.n_labeled) if stats.n_labeled else 0.0

    top_missing = miss_overall.head(10)
    top_impact = impact.head(10)

    def _tbl(df_: pd.DataFrame, cols: List[str]) -> str:
        """Render a small DataFrame slice as a GitHub-flavored Markdown table.

        Avoids `DataFrame.to_markdown()` to keep this script dependency-free.
        """
        view = df_.loc[:, cols].copy()
        # Normalize types for clean display
        for c in cols:
            if view[c].dtype.kind in {"f"}:
                view[c] = view[c].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
            else:
                view[c] = view[c].astype(str).replace({"nan": ""})

        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        rows = ["| " + " | ".join(view.iloc[i].tolist()) + " |" for i in range(len(view))]
        return "\n".join([header, sep] + rows)

    lines = []
    lines.append(f"# Sparsity Report (v6)\n")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")

    lines.append("## Dataset overview\n")
    lines.append(f"- Total rows: {stats.n_rows}\n")
    lines.append(f"- Unique states: {stats.n_states}\n")
    lines.append(f"- Unique districts: {stats.n_districts}\n")
    lines.append(f"- Year range: {stats.year_min}–{stats.year_max}\n")
    lines.append(f"- Labeled rows (`label_outbreak` not NaN): {stats.n_labeled}\n")
    lines.append(f"- Positives among labeled: {stats.n_pos} ({pos_rate:.1f}%)\n")
    lines.append(f"- Engineered feature columns (`feat_*`): {len(feat_cols)}\n")

    lines.append("## Why folds fail (root cause)\n")
    lines.append(
        "Most fold failures are caused by *complete-case filtering* on engineered features. "
        "Many mechanistic/EWS features are undefined early in a district history (rolling windows, 52-week baselines), "
        "so dropping any row with any feature NaN collapses the dataset to almost nothing.\n"
    )
    lines.append(f"- Strict complete-case rows among labeled (require all {len(feat_cols)} features): {strict_all_rows}\n")
    lines.append(
        f"- Strict complete-case rows using CORE feature set (require only {len(core_feat_cols)} features): {core_strict_rows}\n"
    )

    lines.append("## Top missing features (overall)\n")
    lines.append(_tbl(top_missing, ["feature", "overall_missing_pct", "overall_missing_count"]))
    lines.append("\n")

    lines.append("## Top missing features (within labeled rows)\n")
    lines.append(_tbl(miss_labeled.head(10), ["feature", "labeled_missing_pct", "labeled_missing_count"]))
    lines.append("\n")

    lines.append("## Biggest strict-completeness bottlenecks\n")
    lines.append(
        "`delta_if_drop_feature` tells you how many additional labeled rows you would recover "
        "if you *stopped requiring* that single feature to be non-NaN (holding all others fixed).\n"
    )
    lines.append(_tbl(top_impact, ["feature", "labeled_missing", "delta_if_drop_feature"]))
    lines.append("\n")

    lines.append("## Thesis-consistent core feature set (recommended for sparse panels)\n")
    lines.append(
        "Use this for Track A baselines when you need stable CV. It preserves mechanistic + seasonal signal "
        "while avoiding long-baseline EWS features that are mostly undefined in this dataset.\n"
    )
    lines.append("Core set definition (`CORE_FEATURE_SET_V01`):\n")
    lines.append("\n".join([f"- {c}" for c in CORE_FEATURE_SET_V01]))
    lines.append("\n")
    lines.append("### Rationale (what we exclude and why)\n")
    lines.append(
        "- `feat_var_spike_ratio`: requires a long baseline (52 weeks) so it is missing for most district-years in a sparse panel.\n"
        "- Long lags (e.g., `*_lag_8`) and long-window dynamics become undefined when district-year sequences are short or irregular.\n"
        "- LAI lags are frequently missing due to upstream data gaps; treat them as optional or impute carefully.\n"
    )

    out_path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate sparsity report tables")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_default.yaml",
        help="Path to v6 config file (relative to v6 root)",
    )
    args = parser.parse_args()

    v6_root = get_project_root()
    repo_root = get_repo_root()

    cfg = load_config(str(v6_root / args.config))
    features_path = repo_root / cfg["data"]["processed"]["features"]

    out_dir = v6_root / "results" / "analysis" / "sparsity_report"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading engineered features: {features_path}")
    df = pd.read_parquet(features_path)

    # Ensure stable ordering for any group-derived summaries
    if all(c in df.columns for c in ["state", "district", "year", "week"]):
        df = df.sort_values(["state", "district", "year", "week"]).reset_index(drop=True)

    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    stats = compute_dataset_stats(df)

    if "label_outbreak" not in df.columns:
        raise RuntimeError("Expected column 'label_outbreak' not found in features parquet")

    label_df = df[df["label_outbreak"].notna()].copy()

    miss_overall = missingness_table(df, feat_cols + ["label_outbreak"], prefix="overall")
    miss_labeled = missingness_table(label_df, feat_cols, prefix="labeled")

    strict_all_rows = int(len(label_df.dropna(subset=feat_cols)))

    core_feat_cols = select_feature_columns(df.columns, feature_set="core")
    core_strict_rows = int(len(label_df.dropna(subset=core_feat_cols)))

    impact = strict_impact_table(label_df, feat_cols)
    dy = district_year_week_counts(df)

    ly = label_availability_by_year(df)
    ld = label_availability_by_district(df)
    lr = label_missing_reason_counts(df)

    # Save tables
    miss_overall.to_csv(out_dir / "missingness_overall.csv", index=False)
    miss_labeled.to_csv(out_dir / "missingness_labeled.csv", index=False)
    impact.to_csv(out_dir / "feature_strict_impact.csv", index=False)
    if not dy.empty:
        dy.to_csv(out_dir / "district_year_week_counts.csv", index=False)
    if not ly.empty:
        ly.to_csv(out_dir / "label_availability_by_year.csv", index=False)
    if not ld.empty:
        ld.to_csv(out_dir / "label_availability_by_district.csv", index=False)
    if not lr.empty:
        lr.to_csv(out_dir / "label_missing_reason_counts.csv", index=False)

    # Save markdown narrative
    write_markdown_summary(
        out_dir / "SPARSITY_REPORT.md",
        stats=stats,
        feat_cols=feat_cols,
        miss_overall=miss_overall,
        miss_labeled=miss_labeled,
        impact=impact,
        core_feat_cols=core_feat_cols,
        core_strict_rows=core_strict_rows,
        strict_all_rows=strict_all_rows,
    )

    print("\nSaved report to:")
    print(f"  {out_dir}")
    print("\nKey counts:")
    print(f"  labeled_rows={stats.n_labeled}")
    print(f"  strict_all_features_rows={strict_all_rows}")
    print(f"  strict_core_features_rows={core_strict_rows}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
