#!/usr/bin/env python3
"""
Step 2: Build analytic cohort for a project.

Usage:
    python run_step2_cohort.py --project young-adult-2026

This script:
1. Loads raw SQL export CSVs from the project's raw-queries-results/
2. Builds the analytic cohort DataFrame
3. Saves the cohort to analytic-cohort/analytic_cohort.csv
4. Generates and saves Table 1 descriptive statistics
"""

import argparse

from src.config import load_project_config, get_project_paths
from src.cohort_builder import build_analytic_cohort
from src.table_one import describe_cohort, save_table_one


def main():
    parser = argparse.ArgumentParser(
        description="Build analytic cohort for a project"
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

    print(f"Project directory: {paths['project_dir']}")
    print(f"Raw data: {paths['raw']}")

    # Build cohort
    df = build_analytic_cohort(
        raw_dir=paths["raw"],
        config=config,
    )

    # Save analytic cohort
    output_path = paths["analytic_cohort"] / "analytic_cohort.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved analytic cohort: {output_path}")

    # Generate and save Table 1
    print("\nGenerating Table 1...")
    table_one = describe_cohort(df, label="Real")

    use_iso_date = config.get("output", {}).get("use_iso_date", True)
    save_table_one(
        table_one,
        label="real",
        output_dir=paths["analytic_cohort"],
        use_iso_date=use_iso_date,
    )

    print("\nStep 2 complete!")


if __name__ == "__main__":
    main()
