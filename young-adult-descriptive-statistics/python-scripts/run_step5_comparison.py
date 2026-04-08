#!/usr/bin/env python3
"""
Step 5: Compare Table 1 descriptive statistics across real and synthetic cohorts.

Usage:
    python run_step5_comparison.py --project young-adult-2026

This script:
1. Loads real and synthetic cohorts from project directories
2. Generates Table 1 for each dataset
3. Builds a side-by-side comparison
4. Computes fidelity metrics
5. Saves all results to comparison-results/
"""

import argparse
import sys
from pathlib import Path

from src.config import load_project_config, get_project_paths
from src.table_one import describe_cohort, save_table_one
from src.dataset_comparator import (
    load_cohort,
    build_comparison,
    compute_fidelity,
    summarize_fidelity,
    save_comparison_results,
)


def find_latest_file(directory: Path, pattern: str) -> Path | None:
    """Find the most recently modified file matching a glob pattern."""
    files = list(directory.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda f: f.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(
        description="Compare real and synthetic cohorts"
    )
    parser.add_argument(
        "--project",
        required=True,
        help="Project folder name (e.g., young-adult-2026)"
    )
    args = parser.parse_args()

    # Load project configuration
    print(f"Loading project: {args.project}")
    config = load_project_config(args.project)
    paths = get_project_paths(args.project)

    use_iso_date = config.get("output", {}).get("use_iso_date", True)

    # Build file paths
    files = {
        "Real": paths["analytic_cohort"] / "analytic_cohort.csv",
    }

    # Find synthetic files (use latest if multiple exist)
    # Support both naming conventions: synthetic_ctgan_*.csv and synthetic-CTGAN-*.csv
    ctgan_file = find_latest_file(paths["synthetic_data"], "synthetic*[Cc][Tt][Gg][Aa][Nn]*.csv")
    if ctgan_file:
        files["CTGAN"] = ctgan_file

    bayesian_file = find_latest_file(paths["synthetic_data"], "synthetic*[Bb]ayesian*.csv")
    if bayesian_file:
        files["BayesianNet"] = bayesian_file

    # Verify all files exist
    print("\nChecking input files...")
    for label, filepath in files.items():
        if not filepath.exists():
            print(f"  ERROR: {label} file not found: {filepath}")
            sys.exit(1)
        print(f"  {label}: {filepath}")

    if len(files) < 2:
        print("\nWARNING: No synthetic files found. Run step 4 first.")
        print("Continuing with real data only...")

    # Load all datasets
    print("\nLoading datasets...")
    datasets = {}
    for label, filepath in files.items():
        datasets[label] = load_cohort(filepath, label)

    # Generate Table 1 for each
    print("\nGenerating Table 1 for each cohort...")
    tables = {}
    for label, df in datasets.items():
        print(f"\n--- {label} ---")
        tbl = describe_cohort(df, label=label)
        tables[label] = tbl
        save_table_one(
            tbl,
            label=label.lower(),
            output_dir=paths["comparison_results"],
            use_iso_date=use_iso_date,
        )

    # Build side-by-side comparison
    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 80)
    comparison = build_comparison(tables)
    print(comparison.to_string())

    # Fidelity analysis (only if we have synthetic data)
    synth_labels = [label for label in tables.keys() if label != "Real"]
    if synth_labels:
        print("\n" + "=" * 80)
        print("FIDELITY ANALYSIS: Absolute & Relative Differences from Real")
        print("=" * 80)
        fidelity = compute_fidelity(comparison, real_label="Real")

        # Print in a readable format grouped by method
        for method in fidelity["synth_method"].unique():
            print(f"\n--- {method} vs. Real ---")
            method_df = fidelity[fidelity["synth_method"] == method]
            print(method_df[["metric", "real_value", "synth_value", "abs_diff",
                             "rel_diff_pct", "miss_diff_pp"]].to_string(index=False))

        # Summary scores
        print("\n" + "=" * 80)
        print("FIDELITY SUMMARY")
        print("=" * 80)
        summary = summarize_fidelity(fidelity)
        print(summary.to_string(index=False))

        # Save everything
        save_comparison_results(
            comparison=comparison,
            fidelity=fidelity,
            summary=summary,
            output_dir=paths["comparison_results"],
            use_iso_date=use_iso_date,
        )
    else:
        # No synthetic data, just save the comparison
        from datetime import datetime
        date_str = datetime.now().strftime("%Y-%m-%d" if use_iso_date else "%d%b%y")
        comparison_path = paths["comparison_results"] / f"table_one_comparison_{date_str}.csv"
        comparison.to_csv(comparison_path)
        print(f"\nSaved comparison: {comparison_path}")

    print("\nStep 5 complete!")


if __name__ == "__main__":
    main()
