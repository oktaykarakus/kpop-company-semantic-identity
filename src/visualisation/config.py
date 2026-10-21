"""
Project-wide paths for the visualisation package.
"""

from pathlib import Path

# Repository root
ROOT = Path(__file__).resolve().parents[2]

# Outputs
OUTPUTS = ROOT / "outputs"

COMPANY_PROFILES = OUTPUTS / "company_profiles"
COMPANY_SIGNIFICANCE = OUTPUTS / "company_significance"
EXPLAINABILITY = OUTPUTS / "explainability"
TEMPORAL = OUTPUTS / "temporal"
ROBUSTNESS = OUTPUTS / "robustness"
TABLES = OUTPUTS / "tables"

# Figure outputs
FIGURES = OUTPUTS / "figures"

# Manuscript
MANUSCRIPT_FIGURES = ROOT / "paper" / "manuscript" / "figures"

# Company order (used everywhere)
COMPANY_ORDER = [
    "HYBE",
    "JYP",
    "SM",
    "YG",
]