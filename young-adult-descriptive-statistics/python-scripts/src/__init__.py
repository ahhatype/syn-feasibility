"""
MIMIC-IV Young Adult Cohort Analysis Pipeline.

This package provides reusable modules for:
  - config: Project configuration and path resolution
  - table_one: Descriptive statistics (Table 1) generation
  - cohort_builder: Building analytic cohorts from raw query results
  - synthetic_generator: CTGAN and Bayesian Network synthetic data generation
  - dataset_comparator: Comparing real vs. synthetic datasets
"""

from .config import load_project_config, get_project_paths
from .table_one import describe_cohort, save_table_one
