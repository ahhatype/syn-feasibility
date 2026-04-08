-- Follow physionet steps to gain access to MIMIC IV data, 
-- I followed their guidelines for analysis in BigQuery
-- Storing here for memory - sql executed in BigQuery


-- Notes
-- split intentionally to avoid row explosion from many-to-many joins between diagnoses and chartevents
-- v0 of this code was drafted by Claude Code.

-- ============================================
-- QUERY 1: INDEX ADMISSION + PATIENT-LEVEL FIELDS
-- One row per patient (first qualifying admission after real-world 2017)
-- For more about how dates/ages are deidentified: https://physionet.org/content/mimiciv/3.1/
-- MIMIC IV documentaiton for calculating age: https://github.com/MIT-LCP/mimic-iv/blob/master/concepts/demographics/age.sql
-- Table One queries from MIMIC IV tutorial: https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/notebooks/tableone.ipynb
-- Age at admission BETWEEN 20 AND 33'

-- index_adm_table_13Feb26

-- ============================================

WITH qualifying_admissions AS (
  SELECT
    pa.subject_id
    , ad.hadm_id
    , pa.anchor_age
    , pa.anchor_year
    , pa.anchor_year_group -- anchor groups to approximate real age
    , pa.gender -- administrative f/m
    , pa.dod -- date of death
    , ad.admittime
    , ad.dischtime
    , DATETIME_DIFF(ad.admittime, DATETIME(pa.anchor_year, 1, 1, 0, 0, 0), YEAR) + pa.anchor_age AS age_at_admission -- approximate deidentified age
    , EXTRACT(YEAR FROM ad.admittime) - pa.anchor_year AS years_from_anchor -- to approximate real age at admission
    , ROW_NUMBER() OVER (
        PARTITION BY pa.subject_id
        ORDER BY ad.admittime
      ) AS rn -- to identify first qualifying admission per patient
  FROM `physionet-data.mimiciv_3_1_hosp.patients` pa
  INNER JOIN `physionet-data.mimiciv_3_1_hosp.admissions` ad
    ON pa.subject_id = ad.subject_id
  WHERE
    -- Age at admission between 20 and 33
    DATETIME_DIFF(ad.admittime, DATETIME(pa.anchor_year, 1, 1, 0, 0, 0), YEAR) + pa.anchor_age BETWEEN 20 AND 33
    -- Real-world admission year >= 2017
    AND (
      (pa.anchor_year_group = '2020 - 2022') -- ranges here represent the year range of the anchor year groups
      OR (pa.anchor_year_group = '2017 - 2019'
          AND EXTRACT(YEAR FROM ad.admittime) - pa.anchor_year >= 0)
      OR (pa.anchor_year_group = '2014 - 2016'
          AND EXTRACT(YEAR FROM ad.admittime) - pa.anchor_year >= 3) -- conservative approach: ensure ALL are after 2017 (will miss some)
      OR (pa.anchor_year_group = '2011 - 2013'
          AND EXTRACT(YEAR FROM ad.admittime) - pa.anchor_year >= 6)
      OR (pa.anchor_year_group = '2008 - 2010'
          AND EXTRACT(YEAR FROM ad.admittime) - pa.anchor_year >= 9)
    )
)
SELECT
  subject_id
  , hadm_id AS index_hadm_id
  , anchor_age
  , anchor_year
  , anchor_year_group
  , gender
  , dod
  , admittime AS index_admittime
  , dischtime AS index_dischtime
  , age_at_admission
  , years_from_anchor
FROM qualifying_admissions
WHERE rn = 1 -- filter to first qualifying admission per patient
;


-- ============================================
-- QUERY 2: BASELINE ADMISSIONS (5-year lookback from index)
-- This creates a table of all patient records in 5-year lookback and 
-- Multiple rows per patient
-- For: insurance, race — pick closest to index in Python

-- baseline_adm_table_13Feb26

-- ============================================
WITH index_admissions AS (
  SELECT
    pa.subject_id
    , ad.admittime
    , ROW_NUMBER() OVER (
        PARTITION BY pa.subject_id
        ORDER BY ad.admittime
      ) AS rn
  FROM `physionet-data.mimiciv_3_1_hosp.patients` pa
  INNER JOIN `physionet-data.mimiciv_3_1_hosp.admissions` ad
    ON pa.subject_id = ad.subject_id
  WHERE
    DATETIME_DIFF(ad.admittime, DATETIME(pa.anchor_year, 1, 1, 0, 0, 0), YEAR) + pa.anchor_age BETWEEN 20 AND 33
    AND (
      (pa.anchor_year_group = '2020 - 2022')
      OR (pa.anchor_year_group = '2017 - 2019'
          AND EXTRACT(YEAR FROM ad.admittime) - pa.anchor_year >= 0)
      OR (pa.anchor_year_group = '2014 - 2016'
          AND EXTRACT(YEAR FROM ad.admittime) - pa.anchor_year >= 3)
      OR (pa.anchor_year_group = '2011 - 2013'
          AND EXTRACT(YEAR FROM ad.admittime) - pa.anchor_year >= 6)
      OR (pa.anchor_year_group = '2008 - 2010'
          AND EXTRACT(YEAR FROM ad.admittime) - pa.anchor_year >= 9)
    )
)
SELECT
  idx.subject_id
  , idx.admittime AS index_admittime
  , ad.hadm_id AS baseline_hadm_id
  , ad.admittime AS baseline_admittime
  , ad.insurance
  , ad.race
FROM index_admissions idx
INNER JOIN `physionet-data.mimiciv_3_1_hosp.admissions` ad
  ON idx.subject_id = ad.subject_id
  AND ad.admittime <= idx.admittime
  AND ad.admittime >= DATETIME_SUB(idx.admittime, INTERVAL 5 YEAR)
WHERE idx.rn = 1
;


-- ============================================
-- QUERY 3: BASELINE DIAGNOSES (5-year lookback from index)
-- Multiple rows per patient
-- For: T2DM, CVD, BMI-by-ICD flags

-- baseline_dx_table_13Feb26


-- ============================================
WITH index_admissions AS (
  SELECT
    pa.subject_id
    , ad.admittime
    , ROW_NUMBER() OVER (
        PARTITION BY pa.subject_id
        ORDER BY ad.admittime
      ) AS rn
  FROM `physionet-data.mimiciv_3_1_hosp.patients` pa
  INNER JOIN `physionet-data.mimiciv_3_1_hosp.admissions` ad
    ON pa.subject_id = ad.subject_id
  WHERE
    DATETIME_DIFF(ad.admittime, DATETIME(pa.anchor_year, 1, 1, 0, 0, 0), YEAR) + pa.anchor_age BETWEEN 20 AND 33
    AND (
      (pa.anchor_year_group = '2020 - 2022')
      OR (pa.anchor_year_group = '2017 - 2019'
          AND EXTRACT(YEAR FROM ad.admittime) - pa.anchor_year >= 0)
      OR (pa.anchor_year_group = '2014 - 2016'
          AND EXTRACT(YEAR FROM ad.admittime) - pa.anchor_year >= 3)
      OR (pa.anchor_year_group = '2011 - 2013'
          AND EXTRACT(YEAR FROM ad.admittime) - pa.anchor_year >= 6)
      OR (pa.anchor_year_group = '2008 - 2010'
          AND EXTRACT(YEAR FROM ad.admittime) - pa.anchor_year >= 9)
    )
)
SELECT
  idx.subject_id
  , ad_base.hadm_id AS baseline_hadm_id
  , dx.icd_code
  , dx.icd_version
FROM index_admissions idx
INNER JOIN `physionet-data.mimiciv_3_1_hosp.admissions` ad_base
  ON idx.subject_id = ad_base.subject_id
  AND ad_base.admittime <= idx.admittime
  AND ad_base.admittime >= DATETIME_SUB(idx.admittime, INTERVAL 5 YEAR)
INNER JOIN `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` dx
  ON ad_base.hadm_id = dx.hadm_id
WHERE idx.rn = 1
;


-- ============================================
-- QUERY 4: BASELINE HEIGHT & WEIGHT (5-year lookback from index)
-- Multiple rows per patient (ICU-only)
-- For: BMI continuous — pick closest to index in Python
-- itemid: 226730 — Height (cm)
-- itemid: 226512 — Admission Weight (kg)
-- itemid: 224639 — Daily Weight (kg)

-- baseline_icu_chart_table_13Feb26

-- ============================================
WITH index_admissions AS (
  SELECT
    pa.subject_id
    , ad.admittime
    , ROW_NUMBER() OVER (
        PARTITION BY pa.subject_id
        ORDER BY ad.admittime
      ) AS rn
  FROM `physionet-data.mimiciv_3_1_hosp.patients` pa
  INNER JOIN `physionet-data.mimiciv_3_1_hosp.admissions` ad
    ON pa.subject_id = ad.subject_id
  WHERE
    DATETIME_DIFF(ad.admittime, DATETIME(pa.anchor_year, 1, 1, 0, 0, 0), YEAR) + pa.anchor_age BETWEEN 20 AND 33
    AND (
      (pa.anchor_year_group = '2020 - 2022')
      OR (pa.anchor_year_group = '2017 - 2019'
          AND EXTRACT(YEAR FROM ad.admittime) - pa.anchor_year >= 0)
      OR (pa.anchor_year_group = '2014 - 2016'
          AND EXTRACT(YEAR FROM ad.admittime) - pa.anchor_year >= 3)
      OR (pa.anchor_year_group = '2011 - 2013'
          AND EXTRACT(YEAR FROM ad.admittime) - pa.anchor_year >= 6)
      OR (pa.anchor_year_group = '2008 - 2010'
          AND EXTRACT(YEAR FROM ad.admittime) - pa.anchor_year >= 9)
    )
)
SELECT
  idx.subject_id
  , ce.hadm_id
  , ce.itemid
  , ce.charttime
  , ce.valuenum
  , ce.valueuom
FROM index_admissions idx
INNER JOIN `physionet-data.mimiciv_3_1_hosp.admissions` ad_base
  ON idx.subject_id = ad_base.subject_id
  AND ad_base.admittime <= idx.admittime
  AND ad_base.admittime >= DATETIME_SUB(idx.admittime, INTERVAL 5 YEAR)
INNER JOIN `physionet-data.mimiciv_3_1_icu.chartevents` ce
  ON ad_base.hadm_id = ce.hadm_id
WHERE idx.rn = 1
  AND ce.itemid IN (
    226730   -- Height (cm)
    , 226512 -- Admission Weight (kg)
    , 224639 -- Daily Weight (kg)
  )
  AND ce.valuenum IS NOT NULL
;

-- ============================================
-- QUERY 5: BASELINE BMI FROM OMR (5-year lookback)
-- Outpatient-level data, better coverage than chartevents
-- OMR has online medical records to supplement administrative data: https://physionet.org/content/mimiciv/2.1/

-- baseline_omr_table_13Feb26

-- ============================================
WITH index_admissions AS (
  SELECT
    pa.subject_id
    , ad.admittime
    , ROW_NUMBER() OVER (
        PARTITION BY pa.subject_id
        ORDER BY ad.admittime
      ) AS rn
  FROM `physionet-data.mimiciv_3_1_hosp.patients` pa
  INNER JOIN `physionet-data.mimiciv_3_1_hosp.admissions` ad
    ON pa.subject_id = ad.subject_id
  WHERE
    DATETIME_DIFF(ad.admittime, DATETIME(pa.anchor_year, 1, 1, 0, 0, 0), YEAR) + pa.anchor_age BETWEEN 20 AND 33
    AND (
      (pa.anchor_year_group = '2020 - 2022')
      OR (pa.anchor_year_group = '2017 - 2019'
          AND EXTRACT(YEAR FROM ad.admittime) - pa.anchor_year >= 0)
      OR (pa.anchor_year_group = '2014 - 2016'
          AND EXTRACT(YEAR FROM ad.admittime) - pa.anchor_year >= 3)
      OR (pa.anchor_year_group = '2011 - 2013'
          AND EXTRACT(YEAR FROM ad.admittime) - pa.anchor_year >= 6)
      OR (pa.anchor_year_group = '2008 - 2010'
          AND EXTRACT(YEAR FROM ad.admittime) - pa.anchor_year >= 9)
    )
)
SELECT
  idx.subject_id
  , omr.chartdate
  , omr.result_name
  , omr.result_value
FROM index_admissions idx
INNER JOIN `physionet-data.mimiciv_3_1_hosp.omr` omr
  ON idx.subject_id = omr.subject_id
  AND omr.chartdate <= CAST(idx.admittime AS DATE)
  AND omr.chartdate >= DATE_SUB(CAST(idx.admittime AS DATE), INTERVAL 5 YEAR)
WHERE idx.rn = 1
  AND omr.result_name LIKE '%BMI%'
  OR omr.result_name LIKE '%eight%'  -- Height and Weight
;