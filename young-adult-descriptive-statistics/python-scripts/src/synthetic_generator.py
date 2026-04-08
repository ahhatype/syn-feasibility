"""
Synthetic data generation using CTGAN and Bayesian Network methods.

Two methods:
  1. CTGAN (Xu et al., 2019) - Conditional GAN for tabular data via SDV
  2. Bayesian Network with DP (Ping et al., 2017) - via DataSynthesizer

References:
  - Xu, L., et al. (2019). Modeling Tabular Data using Conditional GAN. NeurIPS.
  - Ping, H., et al. (2017). DataSynthesizer: Privacy-Preserving Synthetic Datasets.
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# Columns to synthesize - the analytic features that feed into describe_cohort().
# Excludes identifiers (subject_id, hadm_id), raw datetimes, and intermediate
# fields (anchor_year, days_to_death, etc.) that are structural, not analytic.
DEFAULT_SYNTH_COLUMNS = [
    # Demographics
    "age_at_admission",
    "gender",                     # source column; derive is_female after synthesis
    "insurance",                  # source column; derive is_medicaid after synthesis
    "race_category",
    "ethnicity",
    # Outcomes (from index admission)
    "one_year_mortality",
    "hospital_los_days",
    # Comorbidities (from baseline, boolean)
    "has_t2dm",
    "has_cvd",
    # BMI - continuous measurement (OMR/chartevents, 55% missing)
    "bmi",
    # BMI - ICD-coded categories (from baseline dx)
    "has_bmi_icd_overweight_obese",
    "has_bmi_icd_underweight_normal",
    # Incident conditions (index admission only)
    "has_sepsis",
]


def load_and_prepare(
    input_path: Path,
    synth_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load the analytic cohort and select/prepare columns for synthesis.

    Preparation steps:
      - Select specified columns only
      - Convert boolean columns to int (0/1) for compatibility with generators
      - Preserve NaN in BMI (generators will learn the missingness pattern)

    Parameters
    ----------
    input_path : Path
        Path to the analytic cohort CSV.
    synth_columns : list[str], optional
        Columns to include in synthesis. Defaults to DEFAULT_SYNTH_COLUMNS.

    Returns
    -------
    pd.DataFrame
        Prepared DataFrame for synthesis.
    """
    if synth_columns is None:
        synth_columns = DEFAULT_SYNTH_COLUMNS

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} patients, {df.shape[1]} columns")

    # Select synthesis columns
    df_synth = df[synth_columns].copy()

    # Convert boolean columns to int for generator compatibility
    bool_cols = df_synth.select_dtypes(include=["bool"]).columns.tolist()
    for col in bool_cols:
        df_synth[col] = df_synth[col].astype(int)

    print(f"Prepared {len(df_synth):,} rows, {df_synth.shape[1]} columns for synthesis")
    print(f"Columns: {list(df_synth.columns)}")

    return df_synth


def postprocess_synthetic(df: pd.DataFrame, method_label: str = "SYN") -> pd.DataFrame:
    """
    Reconstruct derived convenience columns needed by describe_cohort().

    These are deterministic transforms of the synthesized source columns:
      - synthetic_id: prefixed with method label (e.g., CTGAN-000001, BN-000001)
      - is_female from gender
      - is_medicaid from insurance
      - bmi_category from bmi (not used in current Table 1, but available)

    Parameters
    ----------
    df : pd.DataFrame
        Raw synthetic data from a generator.
    method_label : str
        Label prefix for synthetic IDs (e.g., "CTGAN", "BN").

    Returns
    -------
    pd.DataFrame
        Postprocessed synthetic data with derived columns.
    """
    df = df.copy()

    # Assign synthetic patient IDs - prefixed by method so origin is always clear
    # and IDs are never confused with real MIMIC subject_ids
    df.insert(0, "synthetic_id", [f"{method_label}-{i:06d}" for i in range(1, len(df) + 1)])

    # Reconstruct is_female
    if "gender" in df.columns:
        df["is_female"] = (df["gender"] == "F").astype(int)

    # Reconstruct is_medicaid (preserving NaN if insurance is missing)
    if "insurance" in df.columns:
        df["is_medicaid"] = np.where(
            df["insurance"].isna(), np.nan,
            np.where(df["insurance"] == "Medicaid", 1, 0)
        )

    # Reconstruct bmi_category from continuous BMI
    if "bmi" in df.columns:
        df["bmi_category"] = pd.cut(
            df["bmi"],
            bins=[0, 18.5, 25.0, 30.0, np.inf],
            labels=["Underweight", "Normal", "Overweight", "Obese"],
            right=False,
        )

    # Ensure boolean flag columns are bool type
    bool_flag_cols = [
        "has_t2dm", "has_cvd",
        "has_bmi_icd_overweight_obese", "has_bmi_icd_underweight_normal",
        "has_sepsis",
    ]
    for col in bool_flag_cols:
        if col in df.columns:
            # Round to 0/1 in case generator produced floats, then cast
            df[col] = df[col].round().clip(0, 1).astype(bool)

    return df


def generate_ctgan(
    df_real: pd.DataFrame,
    n_synthetic: int,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Generate synthetic data using CTGAN (Conditional GAN for tabular data).

    Reference: Xu et al. (2019) "Modeling Tabular Data using Conditional GAN"
    Library: sdv (Synthetic Data Vault)

    CTGAN uses mode-specific normalization for continuous columns and a
    conditional generator to handle imbalanced categorical columns.
    It's the standard deep learning baseline for tabular synthesis.

    Parameters
    ----------
    df_real : pd.DataFrame
        Real data (synthesis columns only).
    n_synthetic : int
        Number of rows to generate.
    config : dict, optional
        Configuration with synthesis parameters.

    Returns
    -------
    pd.DataFrame of synthetic data with same columns as input.
    """
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata

    config = config or {}
    synth_config = config.get("synthesis", {})

    epochs = synth_config.get("ctgan_epochs", 300)
    batch_size = synth_config.get("ctgan_batch_size", 500)
    random_seed = synth_config.get("random_seed", 42)

    print("\n--- CTGAN Synthesis ---")

    # SDV requires metadata describing column types
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_real)

    # Override detected types where needed
    # Boolean flags stored as int may be detected as numerical - force categorical
    bool_cols = [
        "one_year_mortality",
        "has_t2dm", "has_cvd",
        "has_bmi_icd_overweight_obese", "has_bmi_icd_underweight_normal",
        "has_sepsis",
    ]
    for col in bool_cols:
        if col in df_real.columns:
            metadata.update_column(col, sdtype="categorical")

    # Set random seeds for reproducibility
    np.random.seed(random_seed)
    import torch
    torch.manual_seed(random_seed)

    print(f"Training CTGAN model (epochs={epochs}, batch_size={batch_size})...")
    synthesizer = CTGANSynthesizer(
        metadata,
        epochs=epochs,
        batch_size=batch_size,
        verbose=True,
    )
    synthesizer.fit(df_real)

    print(f"Generating {n_synthetic:,} synthetic rows...")
    df_synth = synthesizer.sample(num_rows=n_synthetic)

    print(f"CTGAN complete: {df_synth.shape[0]:,} rows generated")
    return df_synth


def generate_bayesian_net(
    df_real: pd.DataFrame,
    n_synthetic: int,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Generate synthetic data using a Bayesian Network with optional DP.

    Reference: Ping et al. (2017) "DataSynthesizer: Privacy-Preserving
               Synthetic Datasets"
    Library: DataSynthesizer

    Uses a greedy Bayesian network to model attribute correlations, with
    optional differential privacy (epsilon parameter controls the
    privacy-utility tradeoff).

    Parameters
    ----------
    df_real : pd.DataFrame
        Real data (synthesis columns only).
    n_synthetic : int
        Number of rows to generate.
    config : dict, optional
        Configuration with synthesis parameters.

    Returns
    -------
    pd.DataFrame of synthetic data with same columns as input.
    """
    from DataSynthesizer.DataDescriber import DataDescriber
    from DataSynthesizer.DataGenerator import DataGenerator
    import DataSynthesizer.DataGenerator as dg_module

    dg_module.np = np

    config = config or {}
    synth_config = config.get("synthesis", {})

    epsilon = synth_config.get("bayesian_epsilon", 0.1)
    degree = synth_config.get("bayesian_degree", 2)
    random_seed = synth_config.get("random_seed", 42)

    print("\n--- Bayesian Network Synthesis (DataSynthesizer) ---")

    # DataSynthesizer operates on CSV files, so we write to temp files
    with tempfile.TemporaryDirectory() as tmpdir:
        input_csv = os.path.join(tmpdir, "real_data.csv")
        description_json = os.path.join(tmpdir, "description.json")
        output_csv = os.path.join(tmpdir, "synthetic_data.csv")

        df_real.to_csv(input_csv, index=False)
        np.random.seed(random_seed)

        # Build attribute type map
        attribute_to_is_categorical = {}
        for col in df_real.columns:
            if col in ["age_at_admission", "hospital_los_days", "bmi"]:
                attribute_to_is_categorical[col] = False
            else:
                attribute_to_is_categorical[col] = True

        # Describe the dataset (learn the Bayesian network)
        describer = DataDescriber(category_threshold=10)

        print(f"Learning Bayesian network structure (epsilon={epsilon}, degree={degree})...")
        describer.describe_dataset_in_correlated_attribute_mode(
            dataset_file=input_csv,
            epsilon=epsilon,
            k=degree,
            attribute_to_is_categorical=attribute_to_is_categorical,
        )
        describer.save_dataset_description_to_file(description_json)

        # Generate synthetic data
        print(f"Generating {n_synthetic:,} synthetic rows...")
        generator = DataGenerator()
        generator.generate_dataset_in_correlated_attribute_mode(
            n=n_synthetic,
            description_file=description_json,
        )
        generator.save_synthetic_data(output_csv)

        # Load result
        df_synth = pd.read_csv(output_csv)

    print(f"Bayesian Network complete: {df_synth.shape[0]:,} rows generated")
    return df_synth


def save_synthetic(
    df: pd.DataFrame,
    method_label: str,
    output_dir: Path,
    use_iso_date: bool = True,
) -> str:
    """
    Save synthetic dataset to CSV with standardized naming.

    File naming:
      - ISO date (default): synthetic_{method_label}_{YYYY-MM-DD}.csv
      - Legacy date: synthetic-{method_label}-{DDMMMYY}.csv

    Parameters
    ----------
    df : pd.DataFrame
        Synthetic dataset to save.
    method_label : str
        Method name (e.g., "ctgan", "bayesian_net").
    output_dir : Path
        Directory to save into (created if it doesn't exist).
    use_iso_date : bool
        If True, use ISO date format.

    Returns
    -------
    str
        Path to the saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if use_iso_date:
        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = f"synthetic_{method_label.lower()}_{date_str}.csv"
    else:
        date_str = datetime.now().strftime("%d%b%y")
        filename = f"synthetic-{method_label}-{date_str}.csv"

    filepath = output_dir / filename

    df.to_csv(filepath, index=False)
    print(f"  Saved: {filepath}")
    return str(filepath)
