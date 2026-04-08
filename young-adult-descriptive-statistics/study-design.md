# Original Concept

- [x] Identify all patients with any admissions record after 2017 in MIMIC IV that are 20-33 years old as of 2017. Extract all patient record information in a single table enough to do the following tasks.
- [ ] Describe # of unique patients and simple descriptives using pandas frames in python:
    - [ ] (a) Mean age (with SD);
    - [ ] (b) One year mortality, (%, , % missingness) [used NaN for missing (rather than assuming alive) because of the MIMIC-IV censoring issue: dod is censored at 1 year after the patient's last discharge, not the index discharge. So a null dod for a patient with later admissions is ambiguous.]
    - [ ] (c) Hospital length of stay, (mean, % missingness)
    - [ ] (d) Insurance: Medicaid, (%, % missingness)
    - [ ] (e) Diagnosis record for Acute Bronchitis Due To Respiratory Syncytial Virus (RSV) Diagnoses (single ICD 10 code) during Index admittance event
    - [ ] (f) Female Administrative Gender (%, % missingness)
    - [ ] (g) Race (% by category, % missingness) [MIMIC collapses race into standard categories. "UNKNOWN", "UNABLE TO OBTAIN", and "DECLINED" are mapped to NaN]
    - [ ] (h) Ethnicity (% by category, % missingness) [Since MIMIC-IV v2.0+ folds ethnicity into the race column, the script extracts Hispanic/Latino as a separate ethnicity field]
    - [ ] (i) No. patients with ICD-10 codes for T2DM (%)
    - [ ] (j) No. patients with ICD-10 codes for Cerebrovascular Disease (%) 
    - [ ] (k) BMI (continuous, mean, %missingness)
    - [ ] (l) BMI (% by category obesity-overweight-etc., % missingness)
- [ ] Develop a privacy evaluation technnique backed from literature - code this with pandas in python
- [ ] Generate synthesis data using 1-2 python based, literature backed techniques, for the young adult population described in step 1
- [ ] Test the privacy of each synthetic data set using step 3
- [ ] Rerun the descriptive analysis in (2) for each synthetic data set. 
- [ ] Compare fidelity for these results


## Timing

- Index date = first admission (index admission) date on or after 2017
- Age at index admission 
- One year mortality, hospital length of stay calculated from first admission after January 1 2017 rn = 1
- insurance, gender, race, ethnicity, BMI (continuous) calculated from ANY record during BASELINE (on or up to 5 years before index date)
- T2DM, CVD, BMI (by ICD-10 code) also calculated from ANY record during BASELINE (since it's a boolean yes/no which encounter we use doesn't matter.)

# Detailed Steps

## Step 1: Cohort Extraction (BigQuery SQL)

1. Single query joining across MIMIC-IV tables to get one row per admission
2. Filter to cohort (age 20-33 at January 1 2017)
3. Decision: These are the base files for synthetic data generation; avoid derived variables in the direct sql export (except as needed for filter logic; e.g. age, deidentified dates)

### Queries & Results

See file `young-adult-patient-tables.sql`

1. Q1 — Index cohort: One row per patient — demographics, gender, dod, admittime/dischtime for first qualifying admission after 2017 (age 20–33) [`index_adm_table_13Feb26.csv`]
2. Q2 — Baseline admissions: All admissions within 5-year lookback from index — insurance, race (pick closest to index in Python) [`baseline_adm_table_13Feb26.csv`]
3. Q3 — Baseline diagnoses: All ICD codes from admissions within 5-year lookback — for T2DM, CVD, BMI-by-ICD flags [`baseline_dx_table_13Feb26.csv`]
4. Q4 — Baseline height/weight (ICU): Chartevents measurements within 5-year lookback — ICU-only, expect high missingness [`baseline_icu_chart_table_13Feb26.csv`]
5. Q5 — Baseline BMI (OMR): Outpatient BMI/height/weight from OMR table within 5-year lookback — better coverage than chartevents [`baseline_omr_table_13Feb26.csv`]

### Output

Export to CSV/DataFrame — one row per patient, all columns needed for steps 2–7.

## Step 2: Descriptive Statistics (Python/Pandas)

- Missingness: For each variable, report n_missing / n_total. Be explicit that MIMIC-IV's dod only captures in-hospital and SSA-linked deaths, so "one-year mortality" missingness is structural, not random.
- BMI categories: WHO cutoffs (underweight <18.5, normal 18.5–24.9, overweight 25–29.9, obese ≥30)
- Race/ethnicity: MIMIC-IV has known messiness here — multiple categories, inconsistent coding across admissions for the same patient. Decide on a collapsing strategy upfront.

Write this as a reusable function describe_cohort(df) → summary_df

### ICD-10 Codes by Indication (literature backed)

- Cerebrovascular Disease (source: https://vsac.nlm.nih.gov/valueset/2.16.840.1.113762.1.4.1078.1174/expansion/Latest)
- T2DM (source: https://vsac.nlm.nih.gov/valueset/2.16.840.1.113762.1.4.1160.27/expansion/Latest)
- Underweight (source: https://vsac.nlm.nih.gov/valueset/2.16.840.1.113762.1.4.1108.99/expansion/Latest)
- Obese/Overweight: (source: https://vsac.nlm.nih.gov/valueset/2.16.840.1.113762.1.4.1047.501/expansion/Latest)

## Step 3: [strategy in progress; needs more research. Claude suggestion shown here]: Privacy Evaluation (Python)

### Membership Inference Risk / Nearest-Neighbor Distance Ratio

For each record in the synthetic data, find the distance to the nearest record in the real data vs. the nearest other record in the real data
Based on: Stadler et al. (2022), "Synthetic Data — Anonymisation Groundhog Day"; also used in the NIST synthetic data challenges
Implementation: sklearn NearestNeighbors on normalized features

### Attribute Disclosure Risk

Given a set of quasi-identifiers (age, gender, race), how often can you correctly predict a sensitive attribute (diagnosis, insurance) by matching to the synthetic data?
Based on: El Emam et al.'s k-anonymity framework and Taub et al. (2018) on disclosure risk for synthetic data

### Recommended primary metric: 

Distance to Closest Record (DCR) — compute for each real record its distance to the nearest synthetic record, and compare to the distribution of real-to-real distances. If the synthetic data is "too close" to real records, privacy is compromised. This is interpretable and widely cited.

## Step 4: [strategy in progress; needs more research. Claude suggestion shown here]: Synthetic Data Generation (Python)

### SDV (Synthetic Data Vault) — CTGAN or TVAE

Xu et al. (2019), "Modeling Tabular Data using Conditional GAN"
pip install sdv — handles mixed continuous/categorical columns natively
CTGAN uses a conditional GAN architecture designed for tabular data; TVAE is a variational autoencoder alternative
Both are standard benchmarks in the synthetic data literature

### DataSynthesizer (or synthpop-style Bayesian network)

Ping et al. (2017), "DataSynthesizer: Privacy-Preserving Synthetic Datasets"
Uses a differentially-private Bayesian network to model attribute correlations
Useful contrast to CTGAN because it offers explicit differential privacy guarantees
pip install DataSynthesizer

## Step 5: Privacy Testing

- Compute DCR distributions for CTGAN output vs. real data
- Compute DCR distributions for DataSynthesizer output vs. real data
- Compare both to a baseline (real-to-real DCR distribution)
- Report attribute disclosure risk for key sensitive variables

## Step 6: Descriptive Re-analysis

Call describe_cohort() function from Step 2 on each synthetic dataset. 

## Step 7: [strategy in progress; needs more research. Claude suggestion shown here]: Fidelity Comparison

Metrics to compare:

- Marginal distributions: Per-variable comparison (means, proportions) between real and synthetic — you can use standardized mean differences (SMD), which you're familiar with from causal inference
- Pairwise correlations: Compare correlation matrices (real vs. synthetic) — report mean absolute correlation error
- Utility-specific fidelity: How close are the Step 2 descriptives? Present as a side-by-side table: Real | CTGAN | DataSynthesizer

Visualization: A radar chart or grouped bar chart showing each descriptive metric across the three datasets is effective for this.
