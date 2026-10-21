"""
Figure 8 - Integrated Organisational Identity Profiles

Each panel presents one company using the same evidence structure:
    1. Four-dimensional semantic fingerprint profile
       (Exploration, Integration, Breadth, Adaptation)
    2. Relative distinctiveness
    3. Leading semantic communities
    4. Temporal qualification
    5. Comparative context

Input:
    outputs/explainability/integrated_company_identity.csv

Outputs:
    outputs/figures/fig08/
        fig08_integrated_identity_profiles_v1.svg
        fig08_integrated_identity_profiles_v1.pdf
        fig08_integrated_identity_profiles_v1.png
        fig08_integrated_identity_profiles_v1.tiff

    paper/manuscript/figures/
        fig08_integrated_identity_profiles_v1.svg
        fig08_integrated_identity_profiles_v1.pdf
"""

from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image


# ---------------------------------------------------------------------
# Repository imports
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from visualisation.config import (  # noqa: E402
    COMPANY_ORDER,
    EXPLAINABILITY,
    FIGURES,
    MANUSCRIPT_FIGURES,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

IDENTITY_FILE = (
    EXPLAINABILITY
    / "integrated_company_identity.csv"
)

OUTPUT_DIR = FIGURES / "fig08"
OUTPUT_STEM = "fig08_integrated_identity_profiles_v1"

COMPANY_COLOURS = {
    "HYBE": "#3B6FB6",
    "JYP": "#D9822B",
    "SM": "#2A9D8F",
    "YG": "#B6465F",
}

DIMENSIONS = [
    ("exploration_score", "Exploration"),
    ("integration_score", "Integration"),
    ("breadth_score", "Breadth"),
    ("adaptation_score", "Adaptation"),
]

FIGURE_WIDTH = 1320
FIGURE_HEIGHT = 1320
EXPORT_SCALE = 4

SCORE_LIMIT = 2.0


# ---------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """Load and validate the frozen Figure 8 input."""

    if not IDENTITY_FILE.exists():
        raise FileNotFoundError(
            f"Required Figure 8 input was not found:\n{IDENTITY_FILE}"
        )

    identity = pd.read_csv(IDENTITY_FILE)

    required_columns = {
        "company",
        "integrated_semantic_identity",
        "signature_strength",
        "signature_rank",
        "distinctiveness_label",
        "exploration_score",
        "integration_score",
        "breadth_score",
        "adaptation_score",
        "positive_dominant_dimensions",
        "dominant_communities",
        "temporal_profile",
        "stability_index",
        "closest_company",
        "most_differentiated_company",
        "principal_pairwise_contrast",
        "principal_pairwise_direction",
    }

    missing_columns = sorted(
        required_columns.difference(identity.columns)
    )

    if missing_columns:
        raise ValueError(
            "Figure 8 input is missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_columns
            )
        )

    duplicated_companies = identity.loc[
        identity["company"].duplicated(),
        "company",
    ].tolist()

    if duplicated_companies:
        raise ValueError(
            "Duplicate company rows found in Figure 8 input: "
            f"{duplicated_companies}"
        )

    observed_companies = set(identity["company"])
    expected_companies = set(COMPANY_ORDER)

    if observed_companies != expected_companies:
        raise ValueError(
            "Companies in Figure 8 input do not match COMPANY_ORDER.\n"
            f"Expected: {COMPANY_ORDER}\n"
            f"Found: {sorted(observed_companies)}"
        )

    numeric_columns = [
        "signature_strength",
        "signature_rank",
        "exploration_score",
        "integration_score",
        "breadth_score",
        "adaptation_score",
        "stability_index",
    ]

    for column in numeric_columns:
        identity[column] = pd.to_numeric(
            identity[column],
            errors="coerce",
        )

        if identity[column].isna().any():
            invalid_companies = identity.loc[
                identity[column].isna(),
                "company",
            ].tolist()

            raise ValueError(
                f"Missing or non-numeric values found in {column} "
                f"for companies: {invalid_companies}"
            )

    score_columns = [
        column
        for column, _ in DIMENSIONS
    ]

    maximum_absolute_score = float(
        np.abs(
            identity[score_columns].to_numpy(dtype=float)
        ).max()
    )

    if maximum_absolute_score > SCORE_LIMIT:
        raise ValueError(
            "A Figure 8 fingerprint score exceeds the configured "
            f"plotting limit of +/-{SCORE_LIMIT:.1f}. "
            f"Maximum observed absolute score: "
            f"{maximum_absolute_score:.3f}"
        )

    identity = (
        identity.set_index("company")
        .loc[COMPANY_ORDER]
        .reset_index()
    )

    return identity


# ---------------------------------------------------------------------
# Text preparation
# ---------------------------------------------------------------------

def format_identity_title(value: str) -> str:
    """Convert publication-facing identity text to title case."""

    return str(value).strip().title()


def format_communities(value: str) -> str:
    """Compact semicolon-separated community labels."""

    communities = [
        item.strip().replace("Community ", "C")
        for item in str(value).split(";")
        if item.strip()
    ]

    return " / ".join(communities)


def format_dominant_dimensions(value: str) -> str:
    """Format dominant fingerprint dimensions."""

    if pd.isna(value):
        return "No single dominant positive dimension"

    text = str(value).strip()

    if not text or text.lower() == "none":
        return "No single dominant positive dimension"

    dimensions = [
        item.strip().title()
        for item in text.split(";")
        if item.strip()
    ]

    return " + ".join(dimensions)


def create_panel_text(row: pd.Series) -> str:
    """Create the publication-facing evidence summary for one panel."""

    dominant_dimensions = format_dominant_dimensions(
        row["positive_dominant_dimensions"]
    )

    communities = format_communities(
        row["dominant_communities"]
    )

    temporal = str(row["temporal_profile"]).strip()
    temporal = temporal[0].upper() + temporal[1:]

    return (
        f"<b>Distinctiveness:</b> "
        f"{row['distinctiveness_label']} "
        f"(rank {int(row['signature_rank'])}/4; "
        f"strength {row['signature_strength']:.3f})"
        f"<br><b>Dominant dimensions:</b> "
        f"{dominant_dimensions}"
        f"<br><b>Leading communities:</b> "
        f"{communities}"
        f"<br><b>Temporal profile:</b> "
        f"{temporal} "
        f"(stability {row['stability_index']:.1f})"
        f"<br><b>Comparative context:</b> "
        f"closest to {row['closest_company']}; "
        f"most differentiated from "
        f"{row['most_differentiated_company']}"
        f"<br><b>Principal contrast:</b> "
        f"{str(row['principal_pairwise_contrast']).title()} "
        f"({row['principal_pairwise_direction']})"
    )


# ---------------------------------------------------------------------
# Figure construction
# ---------------------------------------------------------------------

def build_figure(identity: pd.DataFrame) -> go.Figure:
    """Construct the final four-panel Figure 8."""

    subplot_titles = [
        (
            f"<b>{row.company}</b>"
            f"<br><span style='font-size:15px'>"
            f"{format_identity_title(row.integrated_semantic_identity)}"
            f"</span>"
        )
        for row in identity.itertuples(index=False)
    ]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.12,
        vertical_spacing=0.2,
    )

    panel_positions = [
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
    ]

    dimension_labels = [
        label
        for _, label in DIMENSIONS
    ]

    for panel_index, (
        row,
        (subplot_row, subplot_col),
    ) in enumerate(
        zip(
            identity.to_dict("records"),
            panel_positions,
        ),
        start=1,
    ):
        company = row["company"]
        scores = [
            float(row[column])
            for column, _ in DIMENSIONS
        ]

        score_text = [
            f"{score:+.2f}"
            for score in scores
        ]

        fig.add_trace(
            go.Bar(
                x=scores,
                y=dimension_labels,
                orientation="h",
                marker={
                    "color": COMPANY_COLOURS[company],
                    "line": {
                        "color": "white",
                        "width": 0.8,
                    },
                },
                text=score_text,
                textposition="outside",
                textfont={
                    "family": "Arial",
                    "size": 13,
                    "color": "black",
                },
                cliponaxis=False,
                customdata=np.column_stack(
                    [
                        [company] * len(scores),
                        score_text,
                    ]
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Dimension: %{y}<br>"
                    "Relative score: %{customdata[1]}"
                    "<extra></extra>"
                ),
                showlegend=False,
            ),
            row=subplot_row,
            col=subplot_col,
        )

        xref = (
            "x domain"
            if panel_index == 1
            else f"x{panel_index} domain"
        )

        yref = (
            "y domain"
            if panel_index == 1
            else f"y{panel_index} domain"
        )

        fig.add_annotation(
            x=0.0,
            y=-0.12,
            xref=xref,
            yref=yref,
            text=create_panel_text(pd.Series(row)),
            showarrow=False,
            align="left",
            xanchor="left",
            yanchor="top",
            font={
                "family": "Arial",
                "size": 13.5,
                "color": "#222222",
            },
            bordercolor="#D9D9D9",
            borderwidth=0.8,
            borderpad=7,
            bgcolor="rgba(248,248,248,0.95)",
        )

        fig.add_vline(
            x=0,
            line_width=1.2,
            line_color="#666666",
            row=subplot_row,
            col=subplot_col,
        )

        fig.update_xaxes(
            range=[-SCORE_LIMIT, SCORE_LIMIT],
            tickvals=[-2, -1, 0, 1, 2],
            tickfont={
                "family": "Arial",
                "size": 12,
            },
            showgrid=True,
            gridcolor="#E8E8E8",
            gridwidth=1,
            zeroline=False,
            fixedrange=True,
            row=subplot_row,
            col=subplot_col,
        )

        fig.update_yaxes(
            autorange="reversed",
            tickfont={
                "family": "Arial",
                "size": 13,
            },
            showgrid=False,
            zeroline=False,
            fixedrange=True,
            row=subplot_row,
            col=subplot_col,
        )

    fig.update_xaxes(
        title={
            "text": "Relative fingerprint score",
            "font": {
                "family": "Arial",
                "size": 14,
            },
        },
        row=2,
        col=1,
    )

    fig.update_xaxes(
        title={
            "text": "Relative fingerprint score",
            "font": {
                "family": "Arial",
                "size": 14,
            },
        },
        row=2,
        col=2,
    )

    fig.update_layout(
        width=FIGURE_WIDTH,
        height=FIGURE_HEIGHT,
        template="simple_white",
        margin={
            "l": 125,
            "r": 70,
            "t": 45,
            "b": 200,
        },
        font={
            "family": "Arial",
            "size": 14,
        },
        paper_bgcolor="white",
        plot_bgcolor="white",
        bargap=0.25,
    )

    # Larger subplot titles, smaller evidence boxes.
    for annotation in fig.layout.annotations:
        if annotation.text and "<b>Distinctiveness:</b>" in annotation.text:
            annotation.font = {
                "family": "Arial",
                "size": 13.5,
                "color": "#222222",
            }
        else:
            annotation.font = {
                "family": "Arial",
                "size": 18,
                "color": "#111111",
            }

    return fig


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------

def export_figure(fig: go.Figure) -> None:
    """Export Figure 8 in vector and high-resolution raster formats."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MANUSCRIPT_FIGURES.mkdir(
        parents=True,
        exist_ok=True,
    )

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

    print("\nFigure 8 exported successfully:")
    print(f"  SVG : {svg_path}")
    print(f"  PDF : {pdf_path}")
    print(f"  PNG : {png_path}")
    print(f"  TIFF: {tiff_path}")
    print(f"  Manuscript directory: {MANUSCRIPT_FIGURES}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    """Generate and export Figure 8."""

    print(f"Project root: {ROOT}")
    print(f"Identity input: {IDENTITY_FILE}")

    identity = load_data()

    print("\nIntegrated organisational identity evidence:")
    print(
        identity[
            [
                "company",
                "integrated_semantic_identity",
                "signature_strength",
                "signature_rank",
                "stability_index",
                "temporal_profile",
                "closest_company",
                "most_differentiated_company",
            ]
        ].to_string(index=False)
    )

    print("\nFour-dimensional fingerprint scores:")
    print(
        identity[
            [
                "company",
                "exploration_score",
                "integration_score",
                "breadth_score",
                "adaptation_score",
            ]
        ].to_string(index=False)
    )

    fig = build_figure(identity=identity)

    export_figure(fig)
    fig.show()


if __name__ == "__main__":
    main()
