"""
Figure 3 - Company Semantic Fingerprints
Panel A: Comparative semantic profile matrix.

Input:
    outputs/company_profiles/tables/
    table_company_semantic_profiles.csv

Outputs:
    outputs/figures/fig03/
        fig03_company_semantic_fingerprints.svg
        fig03_company_semantic_fingerprints.pdf
        fig03_company_semantic_fingerprints.png
        fig03_company_semantic_fingerprints.tiff

    paper/manuscript/figures/
        fig03_company_semantic_fingerprints.svg
        fig03_company_semantic_fingerprints.pdf
"""

from pathlib import Path
import shutil
import sys

import pandas as pd
import plotly.express as px
from PIL import Image


# ---------------------------------------------------------------------
# Repository imports
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from visualisation.config import (  # noqa: E402
    COMPANY_ORDER,
    COMPANY_PROFILES,
    FIGURES,
    MANUSCRIPT_FIGURES,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

INPUT_FILE = (
    COMPANY_PROFILES
    / "tables"
    / "table_company_semantic_profiles.csv"
)

OUTPUT_DIR = FIGURES / "fig03"
OUTPUT_STEM = "fig03_company_semantic_fingerprints"

METRIC_NAMES = {
    "normalised_shannon_entropy": "Semantic Diversity",
    "effective_n_communities": "Effective Communities",
    "simpson_diversity": "Simpson Diversity",
    "hhi_concentration": "Semantic Concentration",
    "dominant_community_share": "Dominant Community Share",
    "top_three_communities_share": "Top-3 Community Share",
    "mean_participation": "Cross-community Participation",
    "mean_neighbour_entropy": "Neighbour Entropy",
    "mean_centroid_novelty": "Centroid Novelty",
    "mean_local_novelty": "Local Novelty",
}

FIGURE_WIDTH = 720
FIGURE_HEIGHT = 520
EXPORT_SCALE = 4


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------

def load_and_validate_data() -> pd.DataFrame:
    """Load and validate the frozen company semantic-profile table."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Figure 3 input file was not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "company",
        *METRIC_NAMES.keys(),
    }

    missing_columns = sorted(required_columns.difference(df.columns))

    if missing_columns:
        raise ValueError(
            "Figure 3 input is missing required columns:\n"
            + "\n".join(f"  - {column}" for column in missing_columns)
        )

    duplicated_companies = df.loc[
        df["company"].duplicated(),
        "company",
    ].tolist()

    if duplicated_companies:
        raise ValueError(
            "Duplicate company rows found in Figure 3 input: "
            f"{duplicated_companies}"
        )

    missing_companies = [
        company
        for company in COMPANY_ORDER
        if company not in set(df["company"])
    ]

    if missing_companies:
        raise ValueError(
            "Figure 3 input is missing companies: "
            f"{missing_companies}"
        )

    df = (
        df.set_index("company")
        .loc[COMPANY_ORDER]
    )

    metric_columns = list(METRIC_NAMES.keys())

    if df[metric_columns].isna().any().any():
        missing_locations = (
            df[metric_columns]
            .isna()
            .stack()
        )

        missing_locations = missing_locations[
            missing_locations
        ].index.tolist()

        raise ValueError(
            "Missing metric values found in Figure 3 input: "
            f"{missing_locations}"
        )

    return df


def prepare_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """Create the compressed row-wise min-max fingerprint matrix."""

    metrics = list(METRIC_NAMES.keys())
    heatmap = df[metrics].T.astype(float)

    row_minimum = heatmap.min(axis=1)
    row_range = heatmap.max(axis=1) - row_minimum

    constant_rows = row_range.eq(0)

    if constant_rows.any():
        constant_metrics = row_range.index[constant_rows].tolist()
        raise ValueError(
            "Cannot normalise metrics with identical values across "
            f"all companies: {constant_metrics}"
        )

    heatmap = heatmap.sub(row_minimum, axis=0)
    heatmap = heatmap.div(row_range, axis=0)

    # Compress the full [0, 1] range to [0.2, 0.8].
    # This preserves relative differences while avoiding excessive contrast.
    heatmap = 0.2 + (0.6 * heatmap)

    heatmap.index = [
        METRIC_NAMES[metric]
        for metric in metrics
    ]

    return heatmap


# ---------------------------------------------------------------------
# Figure construction
# ---------------------------------------------------------------------

def build_figure(heatmap: pd.DataFrame):
    """Construct the publication-ready Plotly heatmap."""

    fig = px.imshow(
        heatmap,
        aspect="auto",
        color_continuous_scale="Teal",
        range_color=(0.2, 0.8),
        text_auto=False,
    )

    fig.update_traces(
        xgap=1,
        ygap=1,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "%{y}<br>"
            "Relative value: %{z:.2f}"
            "<extra></extra>"
        ),
    )

    fig.update_layout(
        width=FIGURE_WIDTH,
        height=FIGURE_HEIGHT,
        template="simple_white",
        coloraxis_showscale=False,
        margin={
            "l": 20,
            "r": 20,
            "t": 60,
            "b": 20,
        },
        font={
            "family": "Arial",
            "size": 16,
        },
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    fig.update_xaxes(
        title=None,
        side="top",
        tickfont={
            "family": "Arial",
            "size": 18,
        },
        showgrid=False,
        zeroline=False,
        fixedrange=True,
    )

    fig.update_yaxes(
        title=None,
        tickfont={
            "family": "Arial",
            "size": 15,
        },
        showgrid=False,
        zeroline=False,
        fixedrange=True,
        autorange="reversed",
    )

    return fig


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------

def export_figure(fig) -> None:
    """Export Figure 3 and copy manuscript-ready vector files."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_FIGURES.mkdir(parents=True, exist_ok=True)

    svg_path = OUTPUT_DIR / f"{OUTPUT_STEM}.svg"
    pdf_path = OUTPUT_DIR / f"{OUTPUT_STEM}.pdf"
    png_path = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
    tiff_path = OUTPUT_DIR / f"{OUTPUT_STEM}.tiff"

    fig.write_image(
        svg_path,
        width=FIGURE_WIDTH,
        height=FIGURE_HEIGHT,
    )

    fig.write_image(
        pdf_path,
        width=FIGURE_WIDTH,
        height=FIGURE_HEIGHT,
    )

    fig.write_image(
        png_path,
        width=FIGURE_WIDTH,
        height=FIGURE_HEIGHT,
        scale=EXPORT_SCALE,
    )

    # Plotly does not export TIFF directly, so convert the high-resolution PNG.
    with Image.open(png_path) as image:
        image.convert("RGB").save(
            tiff_path,
            format="TIFF",
            compression="tiff_lzw",
            dpi=(300, 300),
        )

    shutil.copy2(
        svg_path,
        MANUSCRIPT_FIGURES / svg_path.name,
    )

    shutil.copy2(
        pdf_path,
        MANUSCRIPT_FIGURES / pdf_path.name,
    )

    print("\nFigure 3 exported successfully:")
    print(f"  SVG : {svg_path}")
    print(f"  PDF : {pdf_path}")
    print(f"  PNG : {png_path}")
    print(f"  TIFF: {tiff_path}")
    print(f"  Manuscript directory: {MANUSCRIPT_FIGURES}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    """Generate and export Figure 3."""

    print(f"Project root: {ROOT}")
    print(f"Input file: {INPUT_FILE}")

    df = load_and_validate_data()
    heatmap = prepare_heatmap(df)
    fig = build_figure(heatmap)

    export_figure(fig)

    # Keep this enabled while reviewing figures interactively.
    fig.show()


if __name__ == "__main__":
    main()