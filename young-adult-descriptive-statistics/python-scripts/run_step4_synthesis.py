#!/usr/bin/env python3
"""
Step 4: Generate synthetic data for a project.

Usage:
    python run_step4_synthesis.py --project young-adult-2026
    python run_step4_synthesis.py --project young-adult-2026 --method ctgan
    python run_step4_synthesis.py --project young-adult-2026 --method bayesian

This script:
1. Loads the analytic cohort from analytic-cohort/analytic_cohort.csv
2. Generates synthetic data using CTGAN and/or Bayesian Network
3. Saves synthetic datasets to synthetic-data/
"""

import argparse

from src.config import load_project_config, get_project_paths
from src.synthetic_generator import (
    load_and_prepare,
    generate_ctgan,
    generate_bayesian_net,
    postprocess_synthetic,
    save_synthetic,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic data for a project"
    )
    parser.add_argument(
        "--project",
        required=True,
        help="Project folder name (e.g., young-adult-2026)"
    )
    parser.add_argument(
        "--method",
        choices=["ctgan", "bayesian", "both"],
        default="both",
        help="Synthesis method(s) to run (default: both)"
    )
    parser.add_argument(
        "--n-synthetic",
        type=int,
        default=None,
        help="Number of synthetic rows to generate (default: match input size)"
    )
    args = parser.parse_args()

    # Load project configuration
    print(f"Loading project: {args.project}")
    config = load_project_config(args.project)
    paths = get_project_paths(args.project)

    # Load analytic cohort
    input_path = paths["analytic_cohort"] / "analytic_cohort.csv"
    print(f"\nLoading analytic cohort: {input_path}")
    df_real = load_and_prepare(input_path)

    n_synth = args.n_synthetic or len(df_real)
    print(f"Will generate {n_synth:,} synthetic rows")

    use_iso_date = config.get("output", {}).get("use_iso_date", True)

    # Generate CTGAN
    if args.method in ["ctgan", "both"]:
        df_ctgan_raw = generate_ctgan(df_real, n_synth, config)
        df_ctgan = postprocess_synthetic(df_ctgan_raw, method_label="CTGAN")
        save_synthetic(
            df_ctgan,
            method_label="ctgan",
            output_dir=paths["synthetic_data"],
            use_iso_date=use_iso_date,
        )

    # Generate Bayesian Network
    if args.method in ["bayesian", "both"]:
        df_bayesian_raw = generate_bayesian_net(df_real, n_synth, config)
        df_bayesian = postprocess_synthetic(df_bayesian_raw, method_label="BN")
        save_synthetic(
            df_bayesian,
            method_label="bayesian_net",
            output_dir=paths["synthetic_data"],
            use_iso_date=use_iso_date,
        )

    print("\nStep 4 complete!")


if __name__ == "__main__":
    main()
