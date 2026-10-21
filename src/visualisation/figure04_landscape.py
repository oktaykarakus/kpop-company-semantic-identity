"""
Figure 4 - Semantic Community Landscape and Company Enrichment

Panel A:
    Semantic-community size, measured by the number of Big Four songs.
    Bars are coloured by the dominant company within each community.

Panel B:
    Company-community log2 enrichment.
    Stars mark associations significant after Holm correction
    at alpha = 0.05.

Inputs:
    outputs/explainability/community_semantic_dictionary.csv
    outputs/company_profiles/tables/community_company_concentration.csv
    outputs/company_significance/tables/community_company_enrichment.csv

Outputs:
    outputs/figures/fig04/
        fig04_community_landscape_enrichment.svg
        fig04_community_landscape_enrichment.pdf
        fig04_community_landscape_enrichment.png
        fig04_community_landscape_enrichment.tiff

    paper/manuscript/figures/
        fig04_community_landscape_enrichment.svg
        fig04_community_landscape_enrichment.pdf
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
    COMPANY_PROFILES,
    COMPANY_SIGNIFICANCE,
    EXPLAINABILITY,
    FIGURES,
    MANUSCRIPT_FIGURES,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DICTIONARY_FILE = (
    EXPLAINABILITY
    / "community_semantic_dictionary.csv"
)

CONCENTRATION_FILE = (
    COMPANY_PROFILES
    / "tables"
    / "community_company_concentration.csv"
)

ENRICHMENT_FILE = (
    COMPANY_SIGNIFICANCE
    / "tables"
    / "community_company_enrichment.csv"
)

OUTPUT_DIR = FIGURES / "fig04"
OUTPUT_STEM = "fig04_community_landscape_enrichment"

COMPANY_COLOURS = {
    "HYBE": "#3B6FB6",
    "JYP": "#D9822B",
    "SM": "#2A9D8F",
    "YG": "#B6465F",
}

ENRICHMENT_COLOURSCALE = [
    [0.00, "#2C5D7C"],
    [0.25, "#83B4C9"],
    [0.50, "#F7F7F7"],
    [0.75, "#E59A78"],
    [1.00, "#9D3D3D"],
]

ENRICHMENT_LIMIT = 3.0

FIGURE_WIDTH = 1180
FIGURE_HEIGHT = 780
EXPORT_SCALE = 4


# ---------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------

def load_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load and validate the frozen Figure 4 outputs."""

    required_files = [
        DICTIONARY_FILE,
        CONCENTRATION_FILE,
        ENRICHMENT_FILE,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required Figure 4 input was not found:\n{path}"
            )

    dictionary = pd.read_csv(DICTIONARY_FILE)
    concentration = pd.read_csv(CONCENTRATION_FILE)
    enrichment = pd.read_csv(ENRICHMENT_FILE)

    required_columns = {
        "dictionary": {
            "community",
            "community_evidence_label",
            "n_songs",
        },
        "concentration": {
            "community",
            "n_big4_songs",
            "dominant_company",
            "dominant_company_share",
        },
        "enrichment": {
            "company",
            "community",
            "observed_count",
            "enrichment_ratio",
            "log2_enrichment",
            "holm_adjusted_p",
            "significant_holm_005",
        },
    }

    frames = {
        "dictionary": dictionary,
        "concentration": concentration,
        "enrichment": enrichment,
    }

    for name, frame in frames.items():
        missing_columns = sorted(
            required_columns[name].difference(frame.columns)
        )

        if missing_columns:
            raise ValueError(
                f"{name} input is missing required columns:\n"
                + "\n".join(
                    f"  - {column}"
                    for column in missing_columns
                )
            )

    validate_community_tables(
        dictionary=dictionary,
        concentration=concentration,
    )

    validate_enrichment_table(
        enrichment=enrichment,
        communities=sorted(dictionary["community"].unique()),
    )

    return dictionary, concentration, enrichment


def validate_community_tables(
    dictionary: pd.DataFrame,
    concentration: pd.DataFrame,
) -> None:
    """Validate community-level tables."""

    for name, frame in [
        ("dictionary", dictionary),
        ("concentration", concentration),
    ]:
        duplicated = frame.loc[
            frame["community"].duplicated(),
            "community",
        ].tolist()

        if duplicated:
            raise ValueError(
                f"Duplicate communities found in {name}: "
                f"{duplicated}"
            )

    dictionary_communities = set(dictionary["community"])
    concentration_communities = set(concentration["community"])

    if dictionary_communities != concentration_communities:
        missing_from_concentration = sorted(
            dictionary_communities.difference(
                concentration_communities
            )
        )

        missing_from_dictionary = sorted(
            concentration_communities.difference(
                dictionary_communities
            )
        )

        raise ValueError(
            "Community identifiers do not match across inputs.\n"
            f"Missing from concentration: "
            f"{missing_from_concentration}\n"
            f"Missing from dictionary: "
            f"{missing_from_dictionary}"
        )

    if len(dictionary_communities) != 15:
        raise ValueError(
            "Figure 4 expected 15 semantic communities, "
            f"but found {len(dictionary_communities)}."
        )

    unknown_dominant_companies = sorted(
        set(concentration["dominant_company"])
        .difference(COMPANY_ORDER)
    )

    if unknown_dominant_companies:
        raise ValueError(
            "Unexpected dominant companies found: "
            f"{unknown_dominant_companies}"
        )

    if concentration["n_big4_songs"].isna().any():
        raise ValueError(
            "Missing community-size values found in "
            "n_big4_songs."
        )

    if (
        concentration["n_big4_songs"] <= 0
    ).any():
        invalid_communities = concentration.loc[
            concentration["n_big4_songs"] <= 0,
            "community",
        ].tolist()

        raise ValueError(
            "Non-positive community sizes found for: "
            f"{invalid_communities}"
        )

    invalid_shares = concentration.loc[
        ~concentration["dominant_company_share"].between(
            0,
            1,
            inclusive="both",
        ),
        "community",
    ].tolist()

    if invalid_shares:
        raise ValueError(
            "Dominant-company shares outside [0, 1] "
            f"found for communities: {invalid_shares}"
        )


def validate_enrichment_table(
    enrichment: pd.DataFrame,
    communities: list[int],
) -> None:
    """Validate the complete company-community enrichment grid."""

    duplicated_pairs = (
        enrichment.groupby(
            ["community", "company"],
            dropna=False,
        )
        .size()
    )

    duplicated_pairs = duplicated_pairs[
        duplicated_pairs > 1
    ]

    if not duplicated_pairs.empty:
        raise ValueError(
            "Duplicate company-community enrichment rows found:\n"
            f"{duplicated_pairs}"
        )

    observed_companies = set(enrichment["company"])
    expected_companies = set(COMPANY_ORDER)

    if observed_companies != expected_companies:
        raise ValueError(
            "Companies in the enrichment output do not match "
            "COMPANY_ORDER.\n"
            f"Expected: {COMPANY_ORDER}\n"
            f"Found: {sorted(observed_companies)}"
        )

    observed_pairs = set(
        zip(
            enrichment["community"],
            enrichment["company"],
        )
    )

    expected_pairs = {
        (community, company)
        for community in communities
        for company in COMPANY_ORDER
    }

    missing_pairs = sorted(
        expected_pairs.difference(observed_pairs)
    )

    unexpected_pairs = sorted(
        observed_pairs.difference(expected_pairs)
    )

    if missing_pairs or unexpected_pairs:
        raise ValueError(
            "Incomplete company-community enrichment grid.\n"
            f"Missing pairs: {missing_pairs}\n"
            f"Unexpected pairs: {unexpected_pairs}"
        )

    missing_adjusted_p = enrichment.loc[
        enrichment["holm_adjusted_p"].isna(),
        ["community", "company"],
    ]

    if not missing_adjusted_p.empty:
        raise ValueError(
            "Missing Holm-adjusted p-values found:\n"
            f"{missing_adjusted_p.to_string(index=False)}"
        )

    missing_log2 = enrichment[
        enrichment["log2_enrichment"].isna()
    ]

    invalid_missing_log2 = missing_log2[
        ~(
            missing_log2["observed_count"].eq(0)
            & missing_log2["enrichment_ratio"].eq(0)
        )
    ]

    if not invalid_missing_log2.empty:
        raise ValueError(
            "Unexpected missing log2 enrichment values found:\n"
            f"{invalid_missing_log2.to_string(index=False)}"
        )


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------

def shorten_community_label(
    community: int,
    evidence_label: str,
    n_terms: int = 2,
) -> str:
    """Create a compact but interpretable community label."""

    terms = [
        term.strip()
        for term in str(evidence_label).split("/")
        if term.strip()
    ]

    shortened = " / ".join(terms[:n_terms])

    return f"C{int(community)}  {shortened}"


def prepare_data(
    dictionary: pd.DataFrame,
    concentration: pd.DataFrame,
    enrichment: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Prepare ordered community metadata and enrichment matrices.

    Communities are ordered from largest to smallest using the number
    of Big Four songs. Community ID is used as a deterministic
    secondary ordering criterion.
    """

    community_info = (
        dictionary[
            [
                "community",
                "community_evidence_label",
                "n_songs",
            ]
        ]
        .merge(
            concentration[
                [
                    "community",
                    "n_big4_songs",
                    "dominant_company",
                    "dominant_company_share",
                ]
            ],
            on="community",
            how="inner",
            validate="one_to_one",
        )
    )

    community_info = (
        community_info.sort_values(
            by=["n_big4_songs", "community"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )

    community_info["display_label"] = community_info.apply(
        lambda row: shorten_community_label(
            community=int(row["community"]),
            evidence_label=row["community_evidence_label"],
        ),
        axis=1,
    )

    community_order = community_info["community"].tolist()

    original_enrichment_matrix = (
        enrichment.pivot(
            index="community",
            columns="company",
            values="log2_enrichment",
        )
        .reindex(
            index=community_order,
            columns=COMPANY_ORDER,
        )
    )

    significance_matrix = (
        enrichment.pivot(
            index="community",
            columns="company",
            values="significant_holm_005",
        )
        .reindex(
            index=community_order,
            columns=COMPANY_ORDER,
        )
        .fillna(False)
        .astype(bool)
    )

    adjusted_p_matrix = (
        enrichment.pivot(
            index="community",
            columns="company",
            values="holm_adjusted_p",
        )
        .reindex(
            index=community_order,
            columns=COMPANY_ORDER,
        )
    )

    # Structural zeros correspond to observed_count = 0 and
    # enrichment_ratio = 0. Their mathematical log2 enrichment is
    # negative infinity. For plotting only, they are placed at the
    # lower colour-scale boundary.
    plotting_enrichment_matrix = (
        original_enrichment_matrix.fillna(
            -ENRICHMENT_LIMIT
        )
    )

    return (
        community_info,
        plotting_enrichment_matrix,
        original_enrichment_matrix,
        significance_matrix,
        adjusted_p_matrix,
    )


# ---------------------------------------------------------------------
# Hover data
# ---------------------------------------------------------------------

def create_heatmap_customdata(
    original_enrichment_matrix: pd.DataFrame,
    adjusted_p_matrix: pd.DataFrame,
) -> np.ndarray:
    """Create human-readable enrichment and significance hover data."""

    enrichment_display = original_enrichment_matrix.copy().astype(object)

    for row_index in enrichment_display.index:
        for company in enrichment_display.columns:
            value = enrichment_display.loc[row_index, company]

            if pd.isna(value):
                enrichment_display.loc[
                    row_index,
                    company,
                ] = "−∞ (zero observed)"
            else:
                enrichment_display.loc[
                    row_index,
                    company,
                ] = f"{float(value):.3f}"

    adjusted_p_display = adjusted_p_matrix.copy().astype(object)

    for row_index in adjusted_p_display.index:
        for company in adjusted_p_display.columns:
            value = adjusted_p_display.loc[row_index, company]

            adjusted_p_display.loc[
                row_index,
                company,
            ] = f"{float(value):.4f}"

    return np.dstack(
        [
            enrichment_display.to_numpy(),
            adjusted_p_display.to_numpy(),
        ]
    )


# ---------------------------------------------------------------------
# Figure construction
# ---------------------------------------------------------------------

def build_figure(
    community_info: pd.DataFrame,
    plotting_enrichment_matrix: pd.DataFrame,
    original_enrichment_matrix: pd.DataFrame,
    significance_matrix: pd.DataFrame,
    adjusted_p_matrix: pd.DataFrame,
) -> go.Figure:
    """Construct the final two-panel Figure 4."""

    labels = community_info["display_label"].tolist()

    bar_colours = [
        COMPANY_COLOURS[company]
        for company in community_info["dominant_company"]
    ]

    bar_customdata = np.column_stack(
        [
            community_info["community_evidence_label"],
            community_info["dominant_company"],
            community_info["dominant_company_share"],
        ]
    )

    significance_text = np.where(
        significance_matrix.to_numpy(),
        "★",
        "",
    )

    heatmap_customdata = create_heatmap_customdata(
        original_enrichment_matrix=original_enrichment_matrix,
        adjusted_p_matrix=adjusted_p_matrix,
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.29, 0.71],
        horizontal_spacing=0.055,
    )

    # Panel A: community size
    fig.add_trace(
        go.Bar(
            x=community_info["n_big4_songs"],
            y=labels,
            orientation="h",
            marker={
                "color": bar_colours,
                "line": {
                    "color": "white",
                    "width": 0.8,
                },
            },
            customdata=bar_customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Songs: %{x:.0f}<br>"
                "Dominant company: %{customdata[1]}<br>"
                "Dominant share: %{customdata[2]:.1%}"
                "<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Panel B: enrichment heatmap
    fig.add_trace(
        go.Heatmap(
            z=plotting_enrichment_matrix.to_numpy(),
            x=COMPANY_ORDER,
            y=labels,
            text=significance_text,
            texttemplate="%{text}",
            textfont={
                "family": "Arial",
                "size": 22,
                "color": "black",
            },
            customdata=heatmap_customdata,
            colorscale=ENRICHMENT_COLOURSCALE,
            zmin=-ENRICHMENT_LIMIT,
            zmax=ENRICHMENT_LIMIT,
            zmid=0,
            colorbar={
                "title": {
                    "text": "log₂ enrichment",
                    "side": "right",
                },
                "thickness": 14,
                "len": 0.74,
                "outlinewidth": 0,
                "tickfont": {
                    "family": "Arial",
                    "size": 12,
                },
            },
            xgap=1.2,
            ygap=1.2,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Company: %{x}<br>"
                "log₂ enrichment: %{customdata[0]}<br>"
                "Holm-adjusted p: %{customdata[1]}"
                "<extra></extra>"
            ),
            showscale=True,
        ),
        row=1,
        col=2,
    )

    # Legend-only traces for company colours
    for company in COMPANY_ORDER:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={
                    "size": 10,
                    "color": COMPANY_COLOURS[company],
                    "symbol": "square",
                },
                name=company,
                hoverinfo="skip",
                showlegend=True,
            )
        )

    fig.update_layout(
        width=FIGURE_WIDTH,
        height=FIGURE_HEIGHT,
        template="simple_white",
        margin={
            "l": 220,
            "r": 105,
            "t": 55,
            "b": 75,
        },
        font={
            "family": "Arial",
            "size": 14,
        },
        paper_bgcolor="white",
        plot_bgcolor="white",
        bargap=0.13,
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.075,
            "yanchor": "top",
            "title": None,
            "font": {
                "family": "Arial",
                "size": 13,
            },
            "itemsizing": "constant",
        },
    )

    # Panel labels
    fig.add_annotation(
        x=-0.055,
        y=1.045,
        xref="paper",
        yref="paper",
        text="<b>(A)</b>",
        showarrow=False,
        font={
            "family": "Arial",
            "size": 18,
        },
        xanchor="left",
        yanchor="top",
    )

    fig.add_annotation(
        x=0.315,
        y=1.045,
        xref="paper",
        yref="paper",
        text="<b>(B)</b>",
        showarrow=False,
        font={
            "family": "Arial",
            "size": 18,
        },
        xanchor="left",
        yanchor="top",
    )

    # Panel A axes
    fig.update_xaxes(
        title={
            "text": "Number of songs",
            "font": {
                "family": "Arial",
                "size": 16,
            },
        },
        rangemode="tozero",
        showgrid=True,
        gridcolor="#E6E6E6",
        gridwidth=1,
        zeroline=False,
        tickfont={
            "family": "Arial",
            "size": 13,
        },
        fixedrange=True,
        row=1,
        col=1,
    )

    fig.update_yaxes(
        title=None,
        autorange="reversed",
        tickfont={
            "family": "Arial",
            "size": 12,
        },
        showgrid=False,
        zeroline=False,
        fixedrange=True,
        row=1,
        col=1,
    )

    # Panel B axes
    fig.update_xaxes(
        title=None,
        side="top",
        tickfont={
            "family": "Arial",
            "size": 16,
        },
        showgrid=False,
        zeroline=False,
        fixedrange=True,
        row=1,
        col=2,
    )

    fig.update_yaxes(
        title=None,
        autorange="reversed",
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        fixedrange=True,
        row=1,
        col=2,
    )

    return fig


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------

def export_figure(fig: go.Figure) -> None:
    """Export Figure 4 in vector and high-resolution raster formats."""

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

    print("\nFigure 4 exported successfully:")
    print(f"  SVG : {svg_path}")
    print(f"  PDF : {pdf_path}")
    print(f"  PNG : {png_path}")
    print(f"  TIFF: {tiff_path}")
    print(f"  Manuscript directory: {MANUSCRIPT_FIGURES}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    """Generate and export Figure 4."""

    print(f"Project root: {ROOT}")
    print(f"Dictionary input: {DICTIONARY_FILE}")
    print(f"Concentration input: {CONCENTRATION_FILE}")
    print(f"Enrichment input: {ENRICHMENT_FILE}")

    dictionary, concentration, enrichment = load_data()

    (
        community_info,
        plotting_enrichment_matrix,
        original_enrichment_matrix,
        significance_matrix,
        adjusted_p_matrix,
    ) = prepare_data(
        dictionary=dictionary,
        concentration=concentration,
        enrichment=enrichment,
    )

    print("\nCommunity plotting order:")
    print(
        community_info[
            [
                "community",
                "n_big4_songs",
                "dominant_company",
                "dominant_company_share",
            ]
        ].to_string(index=False)
    )

    fig = build_figure(
        community_info=community_info,
        plotting_enrichment_matrix=plotting_enrichment_matrix,
        original_enrichment_matrix=original_enrichment_matrix,
        significance_matrix=significance_matrix,
        adjusted_p_matrix=adjusted_p_matrix,
    )

    export_figure(fig)
    fig.show()


if __name__ == "__main__":
    main()