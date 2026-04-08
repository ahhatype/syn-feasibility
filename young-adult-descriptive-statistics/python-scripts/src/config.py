"""
Project configuration and path resolution.

Handles loading project-specific configs and resolving paths relative to project folders.
"""

import os
from pathlib import Path
from typing import Any

import yaml


# Root directory of the python-scripts folder
SCRIPTS_ROOT = Path(__file__).parent.parent.resolve()
PROJECTS_DIR = SCRIPTS_ROOT / "projects"
DEFAULT_CONFIG_PATH = SCRIPTS_ROOT / "default_config.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dictionary."""
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_project_config(project_name: str) -> dict[str, Any]:
    """
    Load configuration for a project, merging with defaults.

    Loads default_config.yaml first, then overlays project-specific
    settings from projects/{project_name}/project_config.yaml.

    Parameters
    ----------
    project_name : str
        Name of the project folder (e.g., "young-adult-2026").

    Returns
    -------
    dict
        Merged configuration dictionary.
    """
    # Load defaults
    config = load_yaml(DEFAULT_CONFIG_PATH)

    # Load project-specific config and merge
    project_config_path = PROJECTS_DIR / project_name / "project_config.yaml"
    project_config = load_yaml(project_config_path)

    # Deep merge: project config overrides defaults
    config = _deep_merge(config, project_config)

    # Add computed fields
    config["project_name"] = project_name
    config["project_dir"] = str(PROJECTS_DIR / project_name)

    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def get_project_paths(project_name: str) -> dict[str, Path]:
    """
    Get resolved paths for a project's directories.

    Parameters
    ----------
    project_name : str
        Name of the project folder.

    Returns
    -------
    dict[str, Path]
        Dictionary with keys:
        - project_dir: Root of the project folder
        - raw: raw-queries-results/
        - analytic_cohort: analytic-cohort/
        - synthetic_data: synthetic-data/
        - comparison_results: comparison-results/
    """
    project_dir = PROJECTS_DIR / project_name

    return {
        "project_dir": project_dir,
        "raw": project_dir / "raw-queries-results",
        "analytic_cohort": project_dir / "analytic-cohort",
        "synthetic_data": project_dir / "synthetic-data",
        "comparison_results": project_dir / "comparison-results",
    }


def get_input_file_path(project_name: str, file_key: str) -> Path:
    """
    Get the full path to an input file from the project config.

    Parameters
    ----------
    project_name : str
        Name of the project folder.
    file_key : str
        Key from input_files config (e.g., "index_admissions").

    Returns
    -------
    Path
        Full path to the file.

    Raises
    ------
    KeyError
        If file_key is not found in project config.
    """
    config = load_project_config(project_name)
    paths = get_project_paths(project_name)

    if "input_files" not in config:
        raise KeyError(f"No 'input_files' section in config for project '{project_name}'")

    if file_key not in config["input_files"]:
        raise KeyError(f"No '{file_key}' in input_files for project '{project_name}'")

    filename = config["input_files"][file_key]
    return paths["raw"] / filename
