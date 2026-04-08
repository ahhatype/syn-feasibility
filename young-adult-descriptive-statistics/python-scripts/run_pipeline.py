#!/usr/bin/env python3
"""
Run the full analysis pipeline for a project.

Usage:
    python run_pipeline.py --project young-adult-2026
    python run_pipeline.py --project young-adult-2026 --steps 2,4,5
    python run_pipeline.py --project young-adult-2026 --skip-synthesis

This script runs:
  - Step 2: Build analytic cohort
  - Step 4: Generate synthetic data (CTGAN + Bayesian Network)
  - Step 5: Compare real vs. synthetic cohorts
"""

import argparse
import sys
from pathlib import Path

from src.config import load_project_config, get_project_paths
from src.cohort_builder import build_analytic_cohort
from src.table_one import describe_cohort, save_table_one
from src.synthetic_generator import (
    load_and_prepare,
    generate_ctgan,
    generate_bayesian_net,
    postprocess_synthetic,
    save_synthetic,
)
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


def run_step2(config: dict, paths: dict) -> None:
    """Step 2: Build analytic cohort."""
    print("\n" + "=" * 80)
    print("STEP 2: BUILD ANALYTIC COHORT")
    print("=" * 80)

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


def run_step4(config: dict, paths: dict, method: str = "both") -> None:
    """Step 4: Generate synthetic data."""
    print("\n" + "=" * 80)
    print("STEP 4: GENERATE SYNTHETIC DATA")
    print("=" * 80)

    # Load analytic cohort
    input_path = paths["analytic_cohort"] / "analytic_cohort.csv"
    print(f"\nLoading analytic cohort: {input_path}")
    df_real = load_and_prepare(input_path)

    n_synth = len(df_real)
    print(f"Will generate {n_synth:,} synthetic rows")

    use_iso_date = config.get("output", {}).get("use_iso_date", True)

    # Generate CTGAN
    if method in ["ctgan", "both"]:
        df_ctgan_raw = generate_ctgan(df_real, n_synth, config)
        df_ctgan = postprocess_synthetic(df_ctgan_raw, method_label="CTGAN")
        save_synthetic(
            df_ctgan,
            method_label="ctgan",
            output_dir=paths["synthetic_data"],
            use_iso_date=use_iso_date,
        )

    # Generate Bayesian Network
    if method in ["bayesian", "both"]:
        df_bayesian_raw = generate_bayesian_net(df_real, n_synth, config)
        df_bayesian = postprocess_synthetic(df_bayesian_raw, method_label="BN")
        save_synthetic(
            df_bayesian,
            method_label="bayesian_net",
            output_dir=paths["synthetic_data"],
            use_iso_date=use_iso_date,
        )


def run_step5(config: dict, paths: dict) -> None:
    """Step 5: Compare real and synthetic cohorts."""
    print("\n" + "=" * 80)
    print("STEP 5: COMPARE COHORTS")
    print("=" * 80)

    use_iso_date = config.get("output", {}).get("use_iso_date", True)

    # Build file paths
    files = {
        "Real": paths["analytic_cohort"] / "analytic_cohort.csv",
    }

    # Find synthetic files (support both naming conventions)
    ctgan_file = find_latest_file(paths["synthetic_data"], "synthetic*[Cc][Tt][Gg][Aa][Nn]*.csv")
    if ctgan_file:
        files["CTGAN"] = ctgan_file

    bayesian_file = find_latest_file(paths["synthetic_data"], "synthetic*[Bb]ayesian*.csv")
    if bayesian_file:
        files["BayesianNet"] = bayesian_file

    # Verify real file exists
    if not files["Real"].exists():
        print(f"ERROR: Real cohort not found: {files['Real']}")
        print("Run step 2 first.")
        sys.exit(1)

    # Load all datasets
    print("\nLoading datasets...")
    datasets = {}
    for label, filepath in files.items():
        if filepath.exists():
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
        print("FIDELITY SUMMARY")
        print("=" * 80)
        fidelity = compute_fidelity(comparison, real_label="Real")
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


def main():
    parser = argparse.ArgumentParser(
        description="Run the full analysis pipeline for a project"
    )
    parser.add_argument(
        "--project",
        required=True,
        help="Project folder name (e.g., young-adult-2026)"
    )
    parser.add_argument(
        "--steps",
        default="2,4,5",
        help="Comma-separated list of steps to run (default: 2,4,5)"
    )
    parser.add_argument(
        "--skip-synthesis",
        action="store_true",
        help="Skip step 4 (synthetic data generation)"
    )
    parser.add_argument(
        "--synthesis-method",
        choices=["ctgan", "bayesian", "both"],
        default="both",
        help="Synthesis method for step 4 (default: both)"
    )
    args = parser.parse_args()

    # Parse steps
    steps = [int(s.strip()) for s in args.steps.split(",")]
    if args.skip_synthesis and 4 in steps:
        steps.remove(4)

    # Load project configuration
    print(f"Loading project: {args.project}")
    config = load_project_config(args.project)
    paths = get_project_paths(args.project)

    print(f"Project directory: {paths['project_dir']}")
    print(f"Steps to run: {steps}")

    # Run requested steps
    if 2 in steps:
        run_step2(config, paths)

    if 4 in steps:
        run_step4(config, paths, method=args.synthesis_method)

    if 5 in steps:
        run_step5(config, paths)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
