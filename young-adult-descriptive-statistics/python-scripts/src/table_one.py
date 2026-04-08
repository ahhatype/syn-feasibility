"""
Table 1 descriptive statistics generation.

Provides describe_cohort() for generating Table 1 statistics and
save_table_one() for saving results with standardized naming.
"""

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def describe_cohort(df: pd.DataFrame, label: str = "Real") -> pd.DataFrame:
    """
    Generate Table 1 descriptive statistics for a cohort DataFrame.

    Design decisions:
      - Denominator for race/ethnicity/BMI categories: non-missing count.
        Standard Table 1 convention - percentages reflect the distribution
        among patients with known values only.
      - Denominator for T2DM/CVD: total N. Absence of a diagnosis code
        means "not diagnosed" (False), not "unknown", so no missingness.
      - pct_missing reported for every row (even when 0%) so missingness
        comparison across real vs. synthetic datasets is easy in Step 7.
      - cohort_label column enables pd.concat() + pivot for side-by-side
        comparison across Real / CTGAN / BayesianNet in Step 7.
      - One-year mortality: dod=NaN treated as missing (not alive) because
        MIMIC-IV censors dod at 1yr after LAST discharge, which may differ
        from the index discharge.

    Parameters
    ----------
    df : pd.DataFrame
        Patient-level analytic DataFrame from build_analytic_df().
    label : str
        Label for this cohort (e.g., "Real", "CTGAN", "BayesianNet").

    Returns
    -------
    pd.DataFrame with columns: metric, value, n, pct_missing, cohort_label
    """
    n = len(df)
    rows = []

    def add_row(metric, value, pct_missing=0.0):
        rows.append({
            "metric": metric,
            "value": value,
            "n": n,
            "pct_missing": round(pct_missing, 2),
            "cohort_label": label,
        })

    def pct_miss(series):
        """Percent of values that are NaN."""
        return series.isna().mean() * 100

    # --- N ---
    add_row("N patients", f"{n:,}")

    # --- (a) Age ---
    # No missingness expected: derived from anchor fields, always present.
    age = df["age_at_admission"]
    add_row(
        "Age, mean (SD)",
        f"{age.mean():.1f} ({age.std():.1f})",
        pct_miss(age),
    )

    # --- (b) One-year mortality ---
    # dod censored at 1yr after LAST discharge, not index discharge.
    # dod=NaN could be alive OR censored - treated as missing.
    mort = df["one_year_mortality"]
    mort_valid = mort.dropna()
    n_mort = mort_valid.sum()
    pct_mort = mort_valid.mean() * 100 if len(mort_valid) > 0 else np.nan
    add_row(
        "One-year mortality, n (%)",
        f"{n_mort:,.0f} ({pct_mort:.1f}%)",
        pct_miss(mort),
    )

    # --- (c) Hospital LOS ---
    # Derived from dischtime - admittime; zero missingness expected.
    los = df["hospital_los_days"]
    add_row(
        "Hospital LOS (days), mean (SD)",
        f"{los.mean():.1f} ({los.std():.1f})",
        pct_miss(los),
    )

    # --- (d) Insurance: Medicaid ---
    # From baseline admissions (closest to index). Missing only if no
    # baseline admission had a non-null insurance value.
    ins = df["is_medicaid"]
    ins_valid = ins.dropna()
    n_medicaid = ins_valid.sum()
    pct_medicaid = ins_valid.mean() * 100 if len(ins_valid) > 0 else np.nan
    add_row(
        "Insurance: Medicaid, n (%)",
        f"{n_medicaid:,.0f} ({pct_medicaid:.1f}%)",
        pct_miss(ins),
    )

    # --- (e) Female Administrative Gender ---
    # From patients table - one value per patient, no missingness expected.
    fem = df["is_female"]
    add_row(
        "Female Administrative Gender, n (%)",
        f"{fem.sum():,.0f} ({fem.mean()*100:.1f}%)",
        pct_miss(fem),
    )

    # --- (f) Race (% by category) ---
    # Collapsed from MIMIC-IV race field. "UNKNOWN", "UNABLE TO OBTAIN",
    # "DECLINED" mapped to NaN. Percentages use non-missing denominator.
    race = df["race_category"]
    race_miss = pct_miss(race)
    race_counts = race.value_counts(dropna=True)
    race_n_valid = race.notna().sum()
    for cat in sorted(race_counts.index):
        cnt = race_counts[cat]
        pct = cnt / race_n_valid * 100
        add_row(f"Race: {cat}, n (%)", f"{cnt:,} ({pct:.1f}%)", race_miss)

    # --- (g) Ethnicity (% by category) ---
    # Parsed from MIMIC-IV race field (v2.0+ folds ethnicity into race).
    # Hispanic/Latino extracted; all others = "Not Hispanic/Latino".
    eth = df["ethnicity"]
    eth_miss = pct_miss(eth)
    eth_counts = eth.value_counts(dropna=True)
    eth_n_valid = eth.notna().sum()
    for cat in sorted(eth_counts.index):
        cnt = eth_counts[cat]
        pct = cnt / eth_n_valid * 100
        add_row(f"Ethnicity: {cat}, n (%)", f"{cnt:,} ({pct:.1f}%)", eth_miss)

    # --- (h) T2DM ---
    # Boolean flag from baseline ICD-10 codes (VSAC value set).
    # Denominator = total N: absence = not diagnosed, not missing.
    t2dm = df["has_t2dm"]
    n_t2dm = t2dm.sum()
    add_row(
        "T2DM (ICD-10) [comorbidity, baseline], n (%)",
        f"{n_t2dm:,.0f} ({n_t2dm/n*100:.1f}%)",
    )

    # --- (i) CVD (Cerebrovascular Disease) ---
    # Same logic as T2DM - boolean flag, denominator = total N.
    cvd = df["has_cvd"]
    n_cvd = cvd.sum()
    add_row(
        "CVD (ICD-10) [comorbidity,baseline], n (%)",
        f"{n_cvd:,.0f} ({n_cvd/n*100:.1f}%)",
    )

    # --- (i) Sepsis ---
    # Same logic as T2DM - boolean flag, denominator = total N.
    sepsis = df["has_sepsis"]
    n_sepsis = sepsis.sum()
    add_row(
        "Sepsis (ICD-10) (Single code: A41.9) [incident, index admission only], n (%)",
        f"{n_sepsis:,.0f} ({n_sepsis/n*100:.1f}%)",
    )

    # --- (j) BMI continuous ---
    # From clinical measurements: OMR preferred (better coverage for
    # non-ICU patients), ICU chartevents as fallback.
    # This is a SEPARATE feature from ICD-coded BMI below - different
    # data sources, different coverage, should not be combined.
    bmi = df["bmi"]
    add_row(
        "BMI (continuous), mean (SD)",
        f"{bmi.mean():.1f} ({bmi.std():.1f})",
        pct_miss(bmi),
    )

    # --- (k) BMI by ICD-10 code ---
    # From diagnosis codes in baseline admissions. These are administrative
    # codes assigned by coders, NOT derived from measured height/weight.
    # Denominator = total N (absence of code = not coded, treated as False).
    # Reported as two separate flags since a patient could theoretically
    # have both (e.g., coded normal at one visit, obese at another).
    n_ow_ob = df["has_bmi_icd_overweight_obese"].sum()
    add_row(
        "BMI ICD-10: Overweight/Obese, n (%)",
        f"{n_ow_ob:,.0f} ({n_ow_ob/n*100:.1f}%)",
    )

    n_uw_nm = df["has_bmi_icd_underweight_normal"].sum()
    add_row(
        "BMI ICD-10: Underweight/Normal, n (%)",
        f"{n_uw_nm:,.0f} ({n_uw_nm/n*100:.1f}%)",
    )

    summary = pd.DataFrame(rows)
    return summary


def save_table_one(
    summary: pd.DataFrame,
    label: str,
    output_dir: str | Path = "results",
    use_iso_date: bool = True,
) -> str:
    """
    Save Table 1 results to CSV with standardized naming.

    File naming convention:
      - ISO date (default): table_one_{label}_{YYYY-MM-DD}.csv
      - Legacy date: {label}-table-one-{DDMMMYY}.csv

    Parameters
    ----------
    summary : pd.DataFrame
        Output from describe_cohort().
    label : str
        Cohort label used in filename.
    output_dir : str or Path
        Directory to save into (created if it doesn't exist).
    use_iso_date : bool
        If True, use ISO date format (YYYY-MM-DD). If False, use legacy format.

    Returns
    -------
    str
        Path to the saved file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if use_iso_date:
        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = f"table_one_{label.lower()}_{date_str}.csv"
    else:
        date_str = datetime.now().strftime("%d%b%y")
        filename = f"{label}-table-one-{date_str}.csv"

    filepath = output_dir / filename

    summary.to_csv(filepath, index=False)
    print(f"  Saved: {filepath}")
    return str(filepath)
