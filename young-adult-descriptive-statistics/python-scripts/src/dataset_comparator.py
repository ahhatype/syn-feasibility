"""
Dataset comparison and fidelity analysis.

Compares Table 1 descriptive statistics across real and synthetic cohorts,
computing fidelity metrics to assess how well synthetic data preserves
statistical properties of the real data.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_cohort(filepath: Path, label: str) -> pd.DataFrame:
    """
    Load a cohort CSV and ensure it has the columns describe_cohort() expects.

    For synthetic datasets, some columns may need reconstruction if they
    weren't included in postprocess_synthetic() or if column names differ.

    Parameters
    ----------
    filepath : Path
        Path to the cohort CSV file.
    label : str
        Label for logging (e.g., "Real", "CTGAN").

    Returns
    -------
    pd.DataFrame
        Loaded and prepared cohort data.
    """
    df = pd.read_csv(filepath)
    print(f"  {label}: {len(df):,} rows, {df.shape[1]} columns")

    # Reconstruct derived columns if missing (safety net)
    if "is_female" not in df.columns and "gender" in df.columns:
        df["is_female"] = (df["gender"] == "F").astype(int)

    if "is_medicaid" not in df.columns and "insurance" in df.columns:
        df["is_medicaid"] = np.where(
            df["insurance"].isna(), np.nan,
            np.where(df["insurance"] == "Medicaid", 1, 0)
        )

    if "bmi_category" not in df.columns and "bmi" in df.columns:
        df["bmi_category"] = pd.cut(
            df["bmi"],
            bins=[0, 18.5, 25.0, 30.0, np.inf],
            labels=["Underweight", "Normal", "Overweight", "Obese"],
            right=False,
        )

    # Ensure boolean flag columns are bool
    bool_flag_cols = [
        "has_t2dm", "has_cvd",
        "has_bmi_icd_overweight_obese", "has_bmi_icd_underweight_normal",
        "has_sepsis",
    ]
    for col in bool_flag_cols:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)

    return df


def build_comparison(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine individual Table 1 results into a side-by-side comparison.

    Parameters
    ----------
    tables : dict
        {label: describe_cohort() output DataFrame}

    Returns
    -------
    pd.DataFrame with metric as index, one column per cohort label.
    """
    combined = pd.concat(tables.values(), ignore_index=True)

    # Pivot: rows = metrics, columns = cohort labels
    comparison = combined.pivot(
        index="metric",
        columns="cohort_label",
        values="value",
    )

    # Preserve the original metric order (not alphabetical)
    metric_order = tables[list(tables.keys())[0]]["metric"].tolist()
    comparison = comparison.reindex(metric_order)

    # Add a pct_missing column for each cohort
    for label, tbl in tables.items():
        miss_col = f"pct_missing_{label}"
        miss_map = tbl.set_index("metric")["pct_missing"]
        comparison[miss_col] = comparison.index.map(miss_map)

    # Reorder columns: Real first, then synthetics, then missingness
    cohort_labels = list(tables.keys())
    miss_labels = [f"pct_missing_{label}" for label in cohort_labels]
    comparison = comparison[cohort_labels + miss_labels]

    return comparison


def extract_primary_statistic(value_str: str) -> float:
    """
    Parse the primary statistic from a describe_cohort() formatted value string.

    Parsing rules (based on describe_cohort output formats):
      - "13,112"              -> 13112.0   (N patients: raw count)
      - "27.0 (4.2)"         -> 27.0      (mean/SD: extract mean)
      - "188 (1.4%)"         -> 1.4       (n/%: extract percentage)
      - "4,167 (33.7%)"      -> 33.7      (n/%: extract percentage)

    For comparison purposes, percentages are more meaningful than raw counts
    because synthetic datasets may have different N. Means are compared directly.

    Returns np.nan if parsing fails.
    """
    if pd.isna(value_str) or not isinstance(value_str, str):
        return np.nan

    s = value_str.strip()

    # Pattern: "n (pct%)" - extract the percentage
    if "%" in s:
        try:
            pct_part = s.split("(")[-1].replace("%", "").replace(")", "").strip()
            return float(pct_part)
        except (ValueError, IndexError):
            return np.nan

    # Pattern: "mean (SD)" - extract the mean (no % sign present)
    if "(" in s:
        try:
            mean_part = s.split("(")[0].strip().replace(",", "")
            return float(mean_part)
        except (ValueError, IndexError):
            return np.nan

    # Pattern: plain number (N patients)
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return np.nan


def compute_fidelity(
    comparison: pd.DataFrame,
    real_label: str = "Real",
) -> pd.DataFrame:
    """
    Compute fidelity metrics comparing each synthetic cohort to the real cohort.

    For each metric and synthetic method, computes:
      - real_value: parsed numeric from Real column
      - synth_value: parsed numeric from synthetic column
      - abs_diff: |synth - real|
      - rel_diff_pct: |synth - real| / |real| * 100 (undefined when real = 0)
      - miss_diff_pp: difference in pct_missing (percentage points)

    Parameters
    ----------
    comparison : pd.DataFrame
        Output from build_comparison() with metric as index.
    real_label : str
        Column name for the real cohort.

    Returns
    -------
    pd.DataFrame with fidelity metrics for each synthetic method.
    """
    synth_labels = [
        col for col in comparison.columns
        if col != real_label and not col.startswith("pct_missing_")
    ]

    rows = []
    for metric in comparison.index:
        real_str = comparison.loc[metric, real_label]
        real_val = extract_primary_statistic(real_str)
        real_miss = comparison.loc[metric, f"pct_missing_{real_label}"]

        for synth_label in synth_labels:
            synth_str = comparison.loc[metric, synth_label]
            synth_val = extract_primary_statistic(synth_str)
            synth_miss = comparison.loc[metric, f"pct_missing_{synth_label}"]

            # Absolute difference
            abs_diff = (
                abs(synth_val - real_val)
                if pd.notna(synth_val) and pd.notna(real_val)
                else np.nan
            )

            # Relative difference (% of real value); undefined when real = 0
            if pd.notna(abs_diff) and real_val != 0:
                rel_diff = abs_diff / abs(real_val) * 100
            else:
                rel_diff = np.nan

            # Missingness difference (percentage points)
            miss_diff = (
                (synth_miss - real_miss)
                if pd.notna(synth_miss) and pd.notna(real_miss)
                else np.nan
            )

            rows.append({
                "metric": metric,
                "synth_method": synth_label,
                "real_value": real_val,
                "synth_value": synth_val,
                "abs_diff": round(abs_diff, 3) if pd.notna(abs_diff) else np.nan,
                "rel_diff_pct": round(rel_diff, 2) if pd.notna(rel_diff) else np.nan,
                "miss_diff_pp": round(miss_diff, 2) if pd.notna(miss_diff) else np.nan,
            })

    fidelity = pd.DataFrame(rows)
    return fidelity


def summarize_fidelity(fidelity: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a per-method summary of fidelity across all metrics.

    Reports:
      - mean_abs_diff: average absolute difference across metrics
      - median_abs_diff: median absolute difference (robust to outliers)
      - max_abs_diff: worst-case metric
      - mean_rel_diff_pct: average relative deviation
      - mean_miss_diff_pp: average missingness deviation
      - worst_metric: the metric with the largest absolute difference

    Skips "N patients" row since synthetic N is set to match real N
    and the comparison is uninformative.
    """
    # Exclude N patients from summary - it's a design choice, not a fidelity signal
    df = fidelity[fidelity["metric"] != "N patients"].copy()

    summary_rows = []
    for method, group in df.groupby("synth_method"):
        valid = group.dropna(subset=["abs_diff"])
        worst_idx = valid["abs_diff"].idxmax() if len(valid) > 0 else None

        summary_rows.append({
            "synth_method": method,
            "n_metrics_compared": len(valid),
            "mean_abs_diff": round(valid["abs_diff"].mean(), 3),
            "median_abs_diff": round(valid["abs_diff"].median(), 3),
            "max_abs_diff": round(valid["abs_diff"].max(), 3),
            "worst_metric": valid.loc[worst_idx, "metric"] if worst_idx else None,
            "mean_rel_diff_pct": round(valid["rel_diff_pct"].mean(), 2),
            "mean_miss_diff_pp": round(valid["miss_diff_pp"].mean(), 2),
        })

    return pd.DataFrame(summary_rows)


def save_comparison_results(
    comparison: pd.DataFrame,
    fidelity: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    use_iso_date: bool = True,
) -> dict[str, str]:
    """
    Save all comparison results to CSV files.

    Parameters
    ----------
    comparison : pd.DataFrame
        Side-by-side Table 1 comparison.
    fidelity : pd.DataFrame
        Detailed fidelity metrics.
    summary : pd.DataFrame
        Fidelity summary by method.
    output_dir : Path
        Directory to save into.
    use_iso_date : bool
        If True, use ISO date format in filenames.

    Returns
    -------
    dict[str, str]
        Mapping of result type to saved file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if use_iso_date:
        date_str = datetime.now().strftime("%Y-%m-%d")
    else:
        date_str = datetime.now().strftime("%d%b%y")

    paths = {}

    # Comparison table
    comparison_path = output_dir / f"table_one_comparison_{date_str}.csv"
    comparison.to_csv(comparison_path)
    print(f"Saved comparison: {comparison_path}")
    paths["comparison"] = str(comparison_path)

    # Fidelity detail
    fidelity_path = output_dir / f"fidelity_detail_{date_str}.csv"
    fidelity.to_csv(fidelity_path, index=False)
    print(f"Saved fidelity detail: {fidelity_path}")
    paths["fidelity_detail"] = str(fidelity_path)

    # Fidelity summary
    summary_path = output_dir / f"fidelity_summary_{date_str}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved fidelity summary: {summary_path}")
    paths["fidelity_summary"] = str(summary_path)

    return paths
