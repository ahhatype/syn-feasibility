"""
Cohort builder for creating analysis-ready datasets from raw query results.

Derives variables (age, LOS, one-year mortality, BMI) and produces a single
patient-level analytic dataframe for descriptive analysis.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Suppress FutureWarning for fillna downcasting (pandas >= 2.1)
pd.set_option("future.no_silent_downcasting", True)


# =============================================================================
# ICD-10 CODE DEFINITIONS
# =============================================================================
# Sources: VSAC value sets - see raw-queries-results/icd-10-*.sql for references

# T2DM - Source: https://vsac.nlm.nih.gov/valueset/2.16.840.1.113762.1.4.1160.27/expansion/Latest
CONDITION_T2DM = {
    "E1100", "E1101", "E1110", "E1111", "E1121", "E1122", "E1129",
    "E11311", "E11319", "E11321", "E113211", "E113212", "E113213", "E113219",
    "E11329", "E113291", "E113292", "E113293", "E113299",
    "E11331", "E113311", "E113312", "E113313", "E113319",
    "E11339", "E113391", "E113392", "E113393", "E113399",
    "E11341", "E113411", "E113412", "E113413", "E113419",
    "E11349", "E113491", "E113492", "E113493", "E113499",
    "E11351", "E113511", "E113512", "E113513", "E113519",
    "E113521", "E113522", "E113523", "E113529",
    "E113531", "E113532", "E113533", "E113539",
    "E113541", "E113542", "E113543", "E113549",
    "E113551", "E113552", "E113553", "E113559",
    "E11359", "E113591", "E113592", "E113593", "E113599",
    "E1136", "E1137X1", "E1137X2", "E1137X3", "E1137X9", "E1139",
    "E1140", "E1141", "E1142", "E1143", "E1144", "E1149",
    "E1151", "E1152", "E1159",
    "E11610", "E11618", "E11620", "E11621", "E11622", "E11628",
    "E11630", "E11638", "E11641", "E11649",
    "E1165", "E1169", "E118", "E119",
    "O2412", "O2413",
}

# CVD (Cerebrovascular Disease) - Source: https://vsac.nlm.nih.gov/valueset/2.16.840.1.113762.1.4.1078.1174/expansion/Latest
CONDITION_CVD = {
    "I60", "I600", "I6000", "I6001", "I6002", "I601", "I6010", "I6011", "I6012",
    "I602", "I603", "I6030", "I6031", "I6032", "I604", "I605", "I6050", "I6051",
    "I6052", "I606", "I607", "I608", "I609",
    "I61", "I610", "I611", "I612", "I613", "I614", "I615", "I616", "I618", "I619",
    "I62", "I620", "I6200", "I6201", "I6202", "I6203", "I621", "I629",
    "I63", "I630", "I6300", "I6301", "I63011", "I63012", "I63013", "I63019",
    "I6302", "I6303", "I63031", "I63032", "I63033", "I63039", "I6309",
    "I631", "I6310", "I6311", "I63111", "I63112", "I63113", "I63119",
    "I6312", "I6313", "I63131", "I63132", "I63133", "I63139", "I6319",
    "I632", "I6320", "I6321", "I63211", "I63212", "I63213", "I63219",
    "I6322", "I6323", "I63231", "I63232", "I63233", "I63239", "I6329",
    "I633", "I6330", "I6331", "I63311", "I63312", "I63313", "I63319",
    "I6332", "I63321", "I63322", "I63323", "I63329",
    "I6333", "I63331", "I63332", "I63333", "I63339",
    "I6334", "I63341", "I63342", "I63343", "I63349", "I6339",
    "I634", "I6340", "I6341", "I63411", "I63412", "I63413", "I63419",
    "I6342", "I63421", "I63422", "I63423", "I63429",
    "I6343", "I63431", "I63432", "I63433", "I63439",
    "I6344", "I63441", "I63442", "I63443", "I63449", "I6349",
    "I635", "I6350", "I6351", "I63511", "I63512", "I63513", "I63519",
    "I6352", "I63521", "I63522", "I63523", "I63529",
    "I6353", "I63531", "I63532", "I63533", "I63539",
    "I6354", "I63541", "I63542", "I63543", "I63549", "I6359",
    "I636", "I638", "I6381", "I6389", "I639",
    "I65", "I650", "I6501", "I6502", "I6503", "I6509",
    "I651", "I652", "I6521", "I6522", "I6523", "I6529", "I658", "I659",
    "I66", "I660", "I6601", "I6602", "I6603", "I6609",
    "I661", "I6611", "I6612", "I6613", "I6619",
    "I662", "I6621", "I6622", "I6623", "I6629", "I663", "I668", "I669",
    "I67", "I670", "I671", "I672", "I673", "I674", "I675", "I676", "I677",
    "I678", "I6781", "I6782", "I6783", "I6784", "I67841", "I67848",
    "I6785", "I67850", "I67858", "I6789", "I679",
    "I68", "I680", "I682", "I688",
    "I69", "I690", "I6900", "I6901", "I69010", "I69011", "I69012", "I69013",
    "I69014", "I69015", "I69018", "I69019", "I6902", "I69020", "I69021",
    "I69022", "I69023", "I69028", "I6903", "I69031", "I69032", "I69033",
    "I69034", "I69039", "I6904", "I69041", "I69042", "I69043", "I69044",
    "I69049", "I6905", "I69051", "I69052", "I69053", "I69054", "I69059",
    "I6906", "I69061", "I69062", "I69063", "I69064", "I69065", "I69069",
    "I6909", "I69090", "I69091", "I69092", "I69093", "I69098",
    "I691", "I6910", "I6911", "I69110", "I69111", "I69112", "I69113",
    "I69114", "I69115", "I69118", "I69119", "I6912", "I69120", "I69121",
    "I69122", "I69123", "I69128", "I6913", "I69131", "I69132", "I69133",
    "I69134", "I69139", "I6914", "I69141", "I69142", "I69143", "I69144",
    "I69149", "I6915", "I69151", "I69152", "I69153", "I69154", "I69159",
    "I6916", "I69161", "I69162", "I69163", "I69164", "I69165", "I69169",
    "I6919", "I69190", "I69191", "I69192", "I69193", "I69198",
    "I692", "I6920", "I6921", "I69210", "I69211", "I69212", "I69213",
    "I69214", "I69215", "I69218", "I69219", "I6922", "I69220", "I69221",
    "I69222", "I69223", "I69228", "I6923", "I69231", "I69232", "I69233",
    "I69234", "I69239", "I6924", "I69241", "I69242", "I69243", "I69244",
    "I69249", "I6925", "I69251", "I69252", "I69253", "I69254", "I69259",
    "I6926", "I69261", "I69262", "I69263", "I69264", "I69265", "I69269",
    "I6929", "I69290", "I69291", "I69292", "I69293", "I69298",
    "I693", "I6930", "I6931", "I69310", "I69311", "I69312", "I69313",
    "I69314", "I69315", "I69318", "I69319", "I6932", "I69320", "I69321",
    "I69322", "I69323", "I69328", "I6933", "I69331", "I69332", "I69333",
    "I69334", "I69339", "I6934", "I69341", "I69342", "I69343", "I69344",
    "I69349", "I6935", "I69351", "I69352", "I69353", "I69354", "I69359",
    "I6936", "I69361", "I69362", "I69363", "I69364", "I69365", "I69369",
    "I6939", "I69390", "I69391", "I69392", "I69393", "I69398",
    "I698", "I6980", "I6981", "I69810", "I69811", "I69812", "I69813",
    "I69814", "I69815", "I69818", "I69819", "I6982", "I69820", "I69821",
    "I69822", "I69823", "I69828", "I6983", "I69831", "I69832", "I69833",
    "I69834", "I69839", "I6984", "I69841", "I69842", "I69843", "I69844",
    "I69849", "I6985", "I69851", "I69852", "I69853", "I69854", "I69859",
    "I6986", "I69861", "I69862", "I69863", "I69864", "I69865", "I69869",
    "I6989", "I69890", "I69891", "I69892", "I69893", "I69898",
    "I699", "I6990", "I6991", "I69910", "I69911", "I69912", "I69913",
    "I69914", "I69915", "I69918", "I69919", "I6992", "I69920", "I69921",
    "I69922", "I69923", "I69928", "I6993", "I69931", "I69932", "I69933",
    "I69934", "I69939", "I6994", "I69941", "I69942", "I69943", "I69944",
    "I69949", "I6995", "I69951", "I69952", "I69953", "I69954", "I69959",
    "I6996", "I69961", "I69962", "I69963", "I69964", "I69965", "I69969",
    "I6999", "I69990", "I69991", "I69992", "I69993", "I69998",
}

# Sepsis, from organism unspecified
CONDITION_SEPSIS = {"A419"}

# Obesity/Overweight - Source: https://vsac.nlm.nih.gov/valueset/2.16.840.1.113762.1.4.1047.501/expansion/Latest
CONDITION_OBESE_OVERWEIGHT = {
    "E6601", "E6609", "E661", "E662", "E663",
    "E66811", "E66812", "E66813", "E6689", "E669", "E660",
    "Z6825", "Z6826", "Z6827", "Z6828", "Z6829",
    "Z6830", "Z6831", "Z6832", "Z6833", "Z6834",
    "Z6835", "Z6836", "Z6837", "Z6838", "Z6839",
    "Z684", "Z6841", "Z6842", "Z6843", "Z6844", "Z6845",
}

# Underweight/Normal - Source: https://vsac.nlm.nih.gov/valueset/2.16.840.1.113762.1.4.1108.99/expansion/Latest
CONDITION_UNDERWEIGHT_NORMAL = {
    "R636",
    "Z681", "Z6820", "Z6821", "Z6822", "Z6823", "Z6824",
}

# Combined: all BMI-by-ICD codes
CONDITION_BMI_ICD = CONDITION_OBESE_OVERWEIGHT | CONDITION_UNDERWEIGHT_NORMAL

# ICU chartevents item IDs
ITEMID_HEIGHT_CM = 226730
ITEMID_ADMIT_WEIGHT_KG = 226512
ITEMID_DAILY_WEIGHT_KG = 224639


# =============================================================================
# LOAD RAW DATA
# =============================================================================
def load_data(raw_dir: Path, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """
    Load all CSV files into a dictionary of DataFrames.

    Parameters
    ----------
    raw_dir : Path
        Path to raw-queries-results directory.
    config : dict
        Project configuration with input_files section.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping file keys to DataFrames.
    """
    input_files = config.get("input_files", {})

    file_map = {
        "index": input_files.get("index_admissions"),
        "baseline": input_files.get("baseline_admissions"),
        "dx": input_files.get("baseline_diagnoses"),
        "icu_chart": input_files.get("icu_chartevents"),
        "omr": input_files.get("omr"),
    }

    dfs = {}
    for key, filename in file_map.items():
        if filename is None:
            raise ValueError(f"Missing required input file config: {key}")
        path = raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        df = pd.read_csv(path)
        print(f"  {key}: {df.shape[0]:,} rows, {df.shape[1]} cols")
        dfs[key] = df

    return dfs


# =============================================================================
# DERIVE: INDEX COHORT (Q1)
# =============================================================================
def build_index_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """
    From Q1 results, derive:
      - age_at_admission (already in query as age_at_admission)
      - hospital_los_days
      - one_year_mortality
    """
    df = df.copy()

    # Parse datetimes
    df["index_admittime"] = pd.to_datetime(df["index_admittime"])
    df["index_dischtime"] = pd.to_datetime(df["index_dischtime"])
    df["dod"] = pd.to_datetime(df["dod"], errors="coerce")

    # Deduplicate to one row per patient (first admission = index).
    # Needed if the CSV was exported without the SQL WHERE rn = 1 filter.
    if df["subject_id"].duplicated().any():
        n_before = len(df)
        df.sort_values(["subject_id", "index_admittime"], inplace=True)
        df = df.drop_duplicates(subset=["subject_id"], keep="first")
        print(f"  Deduplicated index: {n_before:,} rows -> {len(df):,} unique patients")

    # Hospital length of stay (fractional days)
    df["hospital_los_days"] = (
        (df["index_dischtime"] - df["index_admittime"]).dt.total_seconds()
        / (24 * 3600)
    )

    # One-year mortality: died within 365 days of index discharge.
    # MIMIC-IV censors dod at 1 year after the patient's LAST discharge.
    # Since last_discharge >= index_discharge (index is the first qualifying
    # admission), NULL dod implies either:
    #   (a) patient is alive, or
    #   (b) patient died >1yr after last discharge, which also means >1yr after index.
    # In both cases, one-year mortality from index = 0. So NULL dod -> 0, not missing.
    df["days_to_death"] = (df["dod"] - df["index_dischtime"]).dt.days
    df["one_year_mortality"] = np.where(
        df["days_to_death"].isna(), 0,  # null dod = survived 1yr from index
        np.where(df["days_to_death"] <= 365, 1, 0)
    )

    return df


# =============================================================================
# DERIVE: BASELINE DEMOGRAPHICS (Q2)
# =============================================================================
def build_baseline_demographics(
    df_baseline: pd.DataFrame, df_index: pd.DataFrame
) -> pd.DataFrame:
    """
    From Q2 results, pick insurance and race closest to index admission.
    Returns one row per patient.
    """
    df = df_baseline.copy()
    df["index_admittime"] = pd.to_datetime(df["index_admittime"])
    df["baseline_admittime"] = pd.to_datetime(df["baseline_admittime"])

    # Time distance from baseline admission to index (always >= 0)
    df["days_to_index"] = (df["index_admittime"] - df["baseline_admittime"]).dt.days

    # For each patient, pick the row closest to index date
    df.sort_values(["subject_id", "days_to_index"], inplace=True)

    # Insurance: closest non-null value
    insurance = (
        df.dropna(subset=["insurance"])
        .groupby("subject_id")
        .first()["insurance"]
        .rename("insurance")
    )

    # Race: closest non-null value
    race = (
        df.dropna(subset=["race"])
        .groupby("subject_id")
        .first()["race"]
        .rename("race")
    )

    demographics = pd.DataFrame({"subject_id": df_index["subject_id"]})
    demographics = demographics.merge(insurance, on="subject_id", how="left")
    demographics = demographics.merge(race, on="subject_id", how="left")

    return demographics


# =============================================================================
# DERIVE: DIAGNOSIS FLAGS (Q3)
# =============================================================================
def build_diagnosis_flags(
    df_dx: pd.DataFrame, df_index: pd.DataFrame
) -> pd.DataFrame:
    """
    From Q3 results, create boolean flags for T2DM, CVD, and BMI-by-ICD.
    Uses exact code sets from VSAC value sets (see CONDITION_* constants).
    Returns one row per patient.
    """
    df = df_dx.copy()

    # Filter to ICD-10 only
    icd10 = df[df["icd_version"] == 10].copy()

    # T2DM: exact match against VSAC value set
    t2dm = (
        icd10[icd10["icd_code"].isin(CONDITION_T2DM)]
        .groupby("subject_id")
        .size()
        .gt(0)
        .rename("has_t2dm")
    )

    # CVD (Cerebrovascular Disease): exact match against VSAC value set
    cvd = (
        icd10[icd10["icd_code"].isin(CONDITION_CVD)]
        .groupby("subject_id")
        .size()
        .gt(0)
        .rename("has_cvd")
    )

    # BMI by ICD: overweight/obese OR underweight/normal codes
    bmi_icd = (
        icd10[icd10["icd_code"].isin(CONDITION_BMI_ICD)]
        .groupby("subject_id")
        .size()
        .gt(0)
        .rename("has_bmi_icd")
    )

    # BMI-by-ICD subcategories for reporting
    bmi_overweight_obese = (
        icd10[icd10["icd_code"].isin(CONDITION_OBESE_OVERWEIGHT)]
        .groupby("subject_id")
        .size()
        .gt(0)
        .rename("has_bmi_icd_overweight_obese")
    )

    bmi_underweight_normal = (
        icd10[icd10["icd_code"].isin(CONDITION_UNDERWEIGHT_NORMAL)]
        .groupby("subject_id")
        .size()
        .gt(0)
        .rename("has_bmi_icd_underweight_normal")
    )

    # Sepsis: index admission ONLY (not full baseline)
    # Filter Q3 dx to only rows matching the index hadm_id
    idx_hadm = df_index[["subject_id", "index_hadm_id"]].copy()
    dx_with_index = icd10.merge(idx_hadm, on="subject_id", how="inner")
    dx_index_only = dx_with_index[
        dx_with_index["baseline_hadm_id"] == dx_with_index["index_hadm_id"]
    ]

    sepsis = (
        dx_index_only[dx_index_only["icd_code"].isin(CONDITION_SEPSIS)]
        .groupby("subject_id")
        .size()
        .gt(0)
        .rename("has_sepsis")
    )

    # Build patient-level flags, default to False (not NaN - absence = no diagnosis)
    flags = pd.DataFrame({"subject_id": df_index["subject_id"]})
    for series in [t2dm, cvd, bmi_icd, bmi_overweight_obese, bmi_underweight_normal, sepsis]:
        flags = flags.merge(series, on="subject_id", how="left")

    # Absence of a diagnosis code = no diagnosis (False), not missing data (NaN).
    # Use astype(bool) after fillna to avoid FutureWarning about silent downcasting.
    flag_cols = [
        "has_t2dm", "has_cvd", "has_bmi_icd",
        "has_bmi_icd_overweight_obese", "has_bmi_icd_underweight_normal",
        "has_sepsis",
    ]
    for col in flag_cols:
        flags[col] = flags[col].fillna(False).astype(bool)

    return flags


# =============================================================================
# DERIVE: BMI FROM ICU CHARTEVENTS (Q4)
# =============================================================================
def build_bmi_from_chartevents(
    df_chart: pd.DataFrame, df_index: pd.DataFrame
) -> pd.DataFrame:
    """
    From Q4 results, compute BMI from height and weight.
    Picks measurement closest to index date for each.
    Returns one row per patient with height_cm, weight_kg, bmi_chart.
    """
    if df_chart.empty:
        result = pd.DataFrame({"subject_id": df_index["subject_id"]})
        result["height_cm_chart"] = np.nan
        result["weight_kg_chart"] = np.nan
        result["bmi_chart"] = np.nan
        return result

    df = df_chart.copy()
    df["charttime"] = pd.to_datetime(df["charttime"])

    # Merge index admittime to compute proximity
    idx_times = df_index[["subject_id", "index_admittime"]].copy()
    idx_times["index_admittime"] = pd.to_datetime(idx_times["index_admittime"])
    df = df.merge(idx_times, on="subject_id", how="left")
    df["days_to_index"] = (df["index_admittime"] - df["charttime"]).dt.days

    # Sort so closest-to-index is first
    df.sort_values(["subject_id", "days_to_index"], inplace=True)

    # Height: closest non-null
    height = (
        df[df["itemid"] == ITEMID_HEIGHT_CM]
        .groupby("subject_id")
        .first()["valuenum"]
        .rename("height_cm_chart")
    )

    # Weight: closest non-null (prefer admission weight, fall back to daily)
    weight_ids = [ITEMID_ADMIT_WEIGHT_KG, ITEMID_DAILY_WEIGHT_KG]
    weight = (
        df[df["itemid"].isin(weight_ids)]
        .groupby("subject_id")
        .first()["valuenum"]
        .rename("weight_kg_chart")
    )

    result = pd.DataFrame({"subject_id": df_index["subject_id"]})
    result = result.merge(height, on="subject_id", how="left")
    result = result.merge(weight, on="subject_id", how="left")

    # BMI = weight (kg) / height (m)^2
    result["bmi_chart"] = result["weight_kg_chart"] / (result["height_cm_chart"] / 100) ** 2

    # Sanity bounds: flag implausible values as NaN
    result.loc[result["bmi_chart"] < 10, "bmi_chart"] = np.nan
    result.loc[result["bmi_chart"] > 80, "bmi_chart"] = np.nan

    return result


# =============================================================================
# DERIVE: BMI FROM OMR (Q5)
# =============================================================================
def build_bmi_from_omr(
    df_omr: pd.DataFrame, df_index: pd.DataFrame
) -> pd.DataFrame:
    """
    From Q5 results, extract BMI closest to index date.
    Returns one row per patient with bmi_omr.
    """
    if df_omr.empty:
        result = pd.DataFrame({"subject_id": df_index["subject_id"]})
        result["bmi_omr"] = np.nan
        return result

    df = df_omr.copy()
    df["chartdate"] = pd.to_datetime(df["chartdate"])

    # Filter to BMI rows (result_name contains 'BMI')
    bmi_rows = df[df["result_name"].str.contains("BMI", case=False, na=False)].copy()

    # Parse numeric BMI value
    bmi_rows["bmi_value"] = pd.to_numeric(bmi_rows["result_value"], errors="coerce")
    bmi_rows = bmi_rows.dropna(subset=["bmi_value"])

    # Merge index admittime to compute proximity
    idx_times = df_index[["subject_id", "index_admittime"]].copy()
    idx_times["index_admittime"] = pd.to_datetime(idx_times["index_admittime"])
    bmi_rows = bmi_rows.merge(idx_times, on="subject_id", how="left")
    bmi_rows["days_to_index"] = (
        bmi_rows["index_admittime"] - pd.to_datetime(bmi_rows["chartdate"])
    ).dt.days

    # Closest to index
    bmi_rows.sort_values(["subject_id", "days_to_index"], inplace=True)
    bmi_closest = bmi_rows.groupby("subject_id").first()["bmi_value"].rename("bmi_omr")

    result = pd.DataFrame({"subject_id": df_index["subject_id"]})
    result = result.merge(bmi_closest, on="subject_id", how="left")

    # Sanity bounds
    result.loc[result["bmi_omr"] < 10, "bmi_omr"] = np.nan
    result.loc[result["bmi_omr"] > 80, "bmi_omr"] = np.nan

    return result


# =============================================================================
# COMBINE: BEST AVAILABLE BMI
# =============================================================================
def combine_bmi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a single bmi column preferring OMR (better coverage)
    over chartevents, plus a WHO category column.
    """
    df = df.copy()
    df["bmi"] = df["bmi_omr"].fillna(df["bmi_chart"])

    # WHO BMI categories
    df["bmi_category"] = pd.cut(
        df["bmi"],
        bins=[0, 18.5, 25.0, 30.0, np.inf],
        labels=["Underweight", "Normal", "Overweight", "Obese"],
        right=False,
    )

    return df


# =============================================================================
# PARSE: RACE AND ETHNICITY
# =============================================================================
def parse_race_ethnicity(df: pd.DataFrame) -> pd.DataFrame:
    """
    MIMIC-IV v2.0+ folds ethnicity into the race column.
    Split Hispanic/Latino out as a separate ethnicity field,
    and collapse race into standard categories.
    """
    df = df.copy()

    # Ethnicity: flag Hispanic/Latino from race field
    df["ethnicity"] = np.where(
        df["race"].str.contains("HISPANIC|LATINO", case=False, na=False),
        "Hispanic/Latino",
        "Not Hispanic/Latino"
    )
    df.loc[df["race"].isna(), "ethnicity"] = np.nan

    # Collapse race into broader categories
    def map_race(val):
        if pd.isna(val):
            return np.nan
        val_upper = val.upper()
        if "WHITE" in val_upper:
            return "White"
        elif "BLACK" in val_upper or "AFRICAN" in val_upper:
            return "Black/African American"
        elif "ASIAN" in val_upper:
            return "Asian"
        elif "HISPANIC" in val_upper or "LATINO" in val_upper:
            return "Hispanic/Latino"
        elif "AMERICAN INDIAN" in val_upper or "ALASKA" in val_upper:
            return "American Indian/Alaska Native"
        elif "NATIVE HAWAIIAN" in val_upper or "PACIFIC" in val_upper:
            return "Native Hawaiian/Pacific Islander"
        elif "UNABLE TO OBTAIN" in val_upper or "UNKNOWN" in val_upper:
            return np.nan
        elif "DECLINED" in val_upper or "NOT SPECIFIED" in val_upper:
            return np.nan
        else:
            return "Other"

    df["race_category"] = df["race"].apply(map_race)

    return df


# =============================================================================
# MAIN: BUILD ANALYTIC COHORT
# =============================================================================
def build_analytic_cohort(raw_dir: Path, config: dict[str, Any]) -> pd.DataFrame:
    """
    Master function: loads all query results and returns a single
    patient-level DataFrame ready for descriptive analysis.

    Parameters
    ----------
    raw_dir : Path
        Path to raw-queries-results directory.
    config : dict
        Project configuration.

    Returns
    -------
    pd.DataFrame
        Patient-level analytic DataFrame.
    """
    print("Loading data...")
    dfs = load_data(raw_dir, config)

    print("\nBuilding index cohort...")
    cohort = build_index_cohort(dfs["index"])

    print("Building baseline demographics...")
    demographics = build_baseline_demographics(dfs["baseline"], cohort)
    demographics = parse_race_ethnicity(demographics)

    print("Building diagnosis flags...")
    dx_flags = build_diagnosis_flags(dfs["dx"], cohort)

    print("Building BMI from chartevents...")
    bmi_chart = build_bmi_from_chartevents(dfs["icu_chart"], cohort)

    print("Building BMI from OMR...")
    bmi_omr = build_bmi_from_omr(dfs["omr"], cohort)

    # Merge everything onto the index cohort
    print("\nMerging all components...")
    analytic = cohort.merge(demographics, on="subject_id", how="left")
    analytic = analytic.merge(dx_flags, on="subject_id", how="left")
    analytic = analytic.merge(
        bmi_chart[["subject_id", "height_cm_chart", "weight_kg_chart", "bmi_chart"]],
        on="subject_id", how="left"
    )
    analytic = analytic.merge(
        bmi_omr[["subject_id", "bmi_omr"]],
        on="subject_id", how="left"
    )

    # Combine BMI sources + categorize
    analytic = combine_bmi(analytic)

    # Convenience: female flag
    analytic["is_female"] = (analytic["gender"] == "F").astype(int)

    # Convenience: Medicaid flag
    analytic["is_medicaid"] = np.where(
        analytic["insurance"].isna(), np.nan,
        np.where(analytic["insurance"] == "Medicaid", 1, 0)
    )

    print(f"\nFinal analytic DataFrame: {analytic.shape[0]:,} patients, {analytic.shape[1]} columns")
    return analytic
