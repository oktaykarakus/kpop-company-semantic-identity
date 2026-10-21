"""
Figure 7 - Temporal Stability, Volatility, and Structural Change

Panel A:
    Ranked organisational stability index for the Big Four entertainment
    companies. Higher values indicate greater temporal stability.

Panel B:
    Decomposition of relative instability into three normalised components:
    mean drift, drift variability, and maximum drift.

Panel C:
    Structural-change profile. Horizontal bars show the detected change-point
    rate, with the raw count and largest detected Jensen-Shannon drift annotated.

Inputs:
    outputs/temporal/rolling5/temporal_stability_summary.csv
    outputs/temporal/rolling5/semantic_change_point_summary.csv

Outputs:
    outputs/figures/fig07/
        fig07_temporal_stability_v2.svg
        fig07_temporal_stability_v2.pdf
        fig07_temporal_stability_v2.png
        fig07_temporal_stability_v2.tiff

    paper/manuscript/figures/
        fig07_temporal_stability_v2.svg
        fig07_temporal_stability_v2.pdf
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

from visualisation import config as visualisation_config  # noqa: E402

COMPANY_ORDER = visualisation_config.COMPANY_ORDER
FIGURES = visualisation_config.FIGURES
MANUSCRIPT_FIGURES = visualisation_config.MANUSCRIPT_FIGURES


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TEMPORAL_ROLLING5 = ROOT / "outputs" / "temporal" / "rolling5"

STABILITY_SUMMARY_FILE = (
    TEMPORAL_ROLLING5 / "temporal_stability_summary.csv"
)

CHANGE_POINT_SUMMARY_FILE = (
    TEMPORAL_ROLLING5 / "semantic_change_point_summary.csv"
)

OUTPUT_DIR = FIGURES / "fig07"
OUTPUT_STEM = "fig07_temporal_stability_v2"

COMPANY_COLOURS = {
    "HYBE": "#3B6FB6",
    "JYP": "#D9822B",
    "SM": "#2A9D8F",
    "YG": "#B6465F",
}

STABILITY_ORDER = ["SM", "YG", "JYP", "HYBE"]

INSTABILITY_COMPONENTS = [
    ("normalised_mean_js", "Mean drift"),
    ("normalised_std_js", "Drift variability"),
    ("normalised_max_js", "Maximum drift"),
]

COMPONENT_COLOURS = {
    "Mean drift": "#4C78A8",
    "Drift variability": "#F28E2B",
    "Maximum drift": "#59A14F",
}

FIGURE_WIDTH = 1650
FIGURE_HEIGHT = 760
EXPORT_SCALE = 4


# ---------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    """Load and validate the frozen Figure 7 temporal outputs."""

    for path in [STABILITY_SUMMARY_FILE, CHANGE_POINT_SUMMARY_FILE]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required Figure 7 input was not found:\n{path}"
            )

    stability = pd.read_csv(STABILITY_SUMMARY_FILE)
    change_points = pd.read_csv(CHANGE_POINT_SUMMARY_FILE)

    required_stability_columns = {
        "company",
        "n_transitions",
        "n_change_points",
        "change_point_rate",
        "normalised_mean_js",
        "normalised_std_js",
        "normalised_max_js",
        "normalised_change_point_rate",
        "relative_instability_score",
        "stability_index",
        "stability_rank",
        "stability_category",
        "largest_change_transition",
        "largest_change_js",
        "largest_change_severity",
    }

    required_change_point_columns = {
        "company",
        "n_transitions",
        "n_change_points",
        "change_point_rate",
        "largest_detected_transition",
        "largest_detected_js",
        "largest_detected_severity",
        "first_change_transition",
        "first_change_from_year",
        "first_change_to_year",
        "change_years",
    }

    missing_stability = sorted(
        required_stability_columns.difference(stability.columns)
    )
    missing_change_points = sorted(
        required_change_point_columns.difference(change_points.columns)
    )

    if missing_stability:
        raise ValueError(
            "temporal_stability_summary.csv is missing required columns:\n"
            + "\n".join(f"  - {column}" for column in missing_stability)
        )

    if missing_change_points:
        raise ValueError(
            "semantic_change_point_summary.csv is missing required columns:\n"
            + "\n".join(f"  - {column}" for column in missing_change_points)
        )

    validate_company_table(stability, "stability summary")
    validate_company_table(change_points, "change-point summary")
    validate_cross_table_consistency(stability, change_points)
    validate_numeric_values(stability, change_points)

    return (
        stability,
        change_points,
        STABILITY_SUMMARY_FILE,
        CHANGE_POINT_SUMMARY_FILE,
    )



def validate_company_table(frame: pd.DataFrame, name: str) -> None:
    """Validate one-row-per-company structure and company coverage."""

    duplicated = frame.loc[
        frame["company"].duplicated(), "company"
    ].tolist()

    if duplicated:
        raise ValueError(f"Duplicate companies found in {name}: {duplicated}")

    observed = set(frame["company"])
    expected = set(COMPANY_ORDER)

    if observed != expected:
        raise ValueError(
            f"Companies in {name} do not match COMPANY_ORDER.\n"
            f"Expected: {COMPANY_ORDER}\n"
            f"Found: {sorted(observed)}"
        )



def validate_cross_table_consistency(
    stability: pd.DataFrame,
    change_points: pd.DataFrame,
) -> None:
    """Check shared structural-change fields across frozen outputs."""

    merged = stability[
        ["company", "n_transitions", "n_change_points", "change_point_rate"]
    ].merge(
        change_points[
            ["company", "n_transitions", "n_change_points", "change_point_rate"]
        ],
        on="company",
        suffixes=("_stability", "_change"),
        validate="one_to_one",
    )

    for field in ["n_transitions", "n_change_points"]:
        mismatch = merged[
            merged[f"{field}_stability"] != merged[f"{field}_change"]
        ]
        if not mismatch.empty:
            raise ValueError(
                f"{field} differs across Figure 7 inputs:\n"
                f"{mismatch.to_string(index=False)}"
            )

    rate_mismatch = merged.loc[
        ~np.isclose(
            merged["change_point_rate_stability"],
            merged["change_point_rate_change"],
            rtol=0.0,
            atol=1e-12,
        )
    ]

    if not rate_mismatch.empty:
        raise ValueError(
            "change_point_rate differs across Figure 7 inputs:\n"
            f"{rate_mismatch.to_string(index=False)}"
        )



def validate_numeric_values(
    stability: pd.DataFrame,
    change_points: pd.DataFrame,
) -> None:
    """Validate the numeric ranges used in Figure 7."""

    stability_numeric = [
        "n_change_points",
        "change_point_rate",
        "normalised_mean_js",
        "normalised_std_js",
        "normalised_max_js",
        "normalised_change_point_rate",
        "relative_instability_score",
        "stability_index",
        "stability_rank",
        "largest_change_js",
    ]

    change_numeric = [
        "n_change_points",
        "change_point_rate",
        "largest_detected_js",
    ]

    for name, frame, columns in [
        ("stability", stability, stability_numeric),
        ("change points", change_points, change_numeric),
    ]:
        if frame[columns].isna().any().any():
            raise ValueError(f"Missing numeric values found in {name} input.")

    for column, _ in INSTABILITY_COMPONENTS:
        invalid = stability.loc[~stability[column].between(0, 1)]
        if not invalid.empty:
            raise ValueError(
                f"{column} contains values outside [0, 1]:\n"
                f"{invalid[['company', column]].to_string(index=False)}"
            )

    invalid_stability = stability.loc[
        ~stability["stability_index"].between(0, 100)
    ]
    if not invalid_stability.empty:
        raise ValueError(
            "stability_index contains values outside [0, 100]:\n"
            f"{invalid_stability[['company', 'stability_index']].to_string(index=False)}"
        )

    invalid_counts = change_points.loc[change_points["n_change_points"] < 0]
    if not invalid_counts.empty:
        raise ValueError(
            "Negative change-point counts found:\n"
            f"{invalid_counts.to_string(index=False)}"
        )


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------


def prepare_data(
    stability: pd.DataFrame,
    change_points: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the frozen stability ranking to both inputs."""

    ordered_stability = (
        stability.set_index("company")
        .reindex(STABILITY_ORDER)
        .reset_index()
    )

    ordered_change_points = (
        change_points.set_index("company")
        .reindex(STABILITY_ORDER)
        .reset_index()
    )

    return ordered_stability, ordered_change_points


# ---------------------------------------------------------------------
# Figure construction
# ---------------------------------------------------------------------


def build_figure(
    stability: pd.DataFrame,
    change_points: pd.DataFrame,
) -> go.Figure:
    """Construct the final three-panel Figure 7."""

    fig = make_subplots(
        rows=1,
        cols=3,
        column_widths=[0.31, 0.42, 0.27],
        horizontal_spacing=0.055,
    )

    # Panel A: ranked stability index
    stability_values = stability["stability_index"].to_numpy(dtype=float)

    fig.add_trace(
        go.Bar(
            x=stability_values,
            y=stability["company"],
            orientation="h",
            marker={
                "color": [
                    COMPANY_COLOURS[company]
                    for company in stability["company"]
                ],
                "line": {"color": "white", "width": 0.8},
            },
            customdata=np.column_stack(
                [
                    stability["stability_rank"],
                    stability["stability_category"],
                    stability["relative_instability_score"],
                ]
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Stability index: %{x:.1f}<br>"
                "Rank: %{customdata[0]:.0f}<br>"
                "Category: %{customdata[1]}<br>"
                "Relative instability: %{customdata[2]:.3f}"
                "<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    for row in stability.itertuples(index=False):
        fig.add_annotation(
            x=min(float(row.stability_index) + 2.2, 103.0),
            y=row.company,
            xref="x",
            yref="y",
            text=str(row.stability_category).replace("_", " ").title(),
            showarrow=False,
            xanchor="left",
            font={"family": "Arial", "size": 14, "color": "#444444"},
        )

    # Panel B: normalised instability decomposition
    for column, label in INSTABILITY_COMPONENTS:
        fig.add_trace(
            go.Bar(
                x=stability[column],
                y=stability["company"],
                orientation="h",
                name=label,
                marker={
                    "color": COMPONENT_COLOURS[label],
                    "line": {"color": "white", "width": 0.6},
                },
                customdata=np.column_stack(
                    [
                        stability["stability_index"],
                        stability["relative_instability_score"],
                    ]
                ),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    + f"{label}: %{{x:.3f}}<br>"
                    + "Stability index: %{customdata[0]:.1f}<br>"
                    + "Relative instability: %{customdata[1]:.3f}"
                    + "<extra></extra>"
                ),
                showlegend=True,
            ),
            row=1,
            col=2,
        )

    # Panel C: detected structural changes
    rates = change_points["change_point_rate"].to_numpy(dtype=float)

    fig.add_trace(
        go.Bar(
            x=rates,
            y=change_points["company"],
            orientation="h",
            marker={
                "color": [
                    COMPANY_COLOURS[company]
                    for company in change_points["company"]
                ],
                "line": {"color": "white", "width": 0.8},
            },
            customdata=np.column_stack(
                [
                    change_points["n_change_points"],
                    change_points["largest_detected_js"],
                    change_points["largest_detected_transition"],
                    change_points["largest_detected_severity"],
                    change_points["change_years"],
                ]
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Change-point rate: %{x:.3f}<br>"
                "Detected change points: %{customdata[0]:.0f}<br>"
                "Largest detected JS: %{customdata[1]:.3f}<br>"
                "Largest transition: %{customdata[2]}<br>"
                "Severity: %{customdata[3]}<br>"
                "Change years: %{customdata[4]}"
                "<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=3,
    )

    for row in change_points.itertuples(index=False):
        fig.add_annotation(
            x=float(row.change_point_rate) + 0.008,
            y=row.company,
            xref="x3",
            yref="y3",
            text=(
                f"n = {int(row.n_change_points)};<br>"
                f"max JS = {float(row.largest_detected_js):.3f}"
            ),
            showarrow=False,
            xanchor="left",
            font={"family": "Arial", "size": 14, "color": "#444444"},
        )

    fig.update_layout(
        width=FIGURE_WIDTH,
        height=FIGURE_HEIGHT,
        template="simple_white",
        barmode="group",
        bargap=0.22,
        bargroupgap=0.06,
        margin={"l": 95, "r": 55, "t": 76, "b": 112},
        font={"family": "Arial", "size": 17},
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend={
            "orientation": "h",
            "x": 0.54,
            "xanchor": "center",
            "y": -0.15,
            "yanchor": "top",
            "title": None,
            "font": {"family": "Arial", "size": 15},
            "itemsizing": "constant",
        },
    )

    panel_positions = [
        (-0.02, "<b>(A)</b>"),
        (0.35, "<b>(B)</b>"),
        (0.78, "<b>(C)</b>"),
    ]

    for x_position, label in panel_positions:
        fig.add_annotation(
            x=x_position,
            y=1.065,
            xref="paper",
            yref="paper",
            text=label,
            showarrow=False,
            font={"family": "Arial", "size": 22},
            xanchor="left",
            yanchor="top",
        )

    common_yaxis = {
        "title": None,
        "categoryorder": "array",
        "categoryarray": STABILITY_ORDER,
        "autorange": "reversed",
        "showgrid": False,
        "zeroline": False,
        "tickfont": {"family": "Arial", "size": 17},
        "fixedrange": True,
    }

    # Panel A axes
    fig.update_xaxes(
        title={
            "text": "Stability index",
            "font": {"family": "Arial", "size": 21},
        },
        range=[0, 115],
        dtick=20,
        showgrid=True,
        gridcolor="#E6E6E6",
        gridwidth=1,
        zeroline=False,
        tickfont={"family": "Arial", "size": 16},
        fixedrange=True,
        row=1,
        col=1,
    )
    fig.update_yaxes(**common_yaxis, row=1, col=1)

    # Panel B axes
    fig.update_xaxes(
        title={
            "text": "Normalised instability component",
            "font": {"family": "Arial", "size": 19},
        },
        range=[0, 1.04],
        dtick=0.2,
        tickformat=".1f",
        showgrid=True,
        gridcolor="#E6E6E6",
        gridwidth=1,
        zeroline=False,
        tickfont={"family": "Arial", "size": 16},
        fixedrange=True,
        row=1,
        col=2,
    )
    fig.update_yaxes(**common_yaxis, row=1, col=2)

    # Panel C axes
    fig.update_xaxes(
        title={
            "text": "Change-point rate",
            "font": {"family": "Arial", "size": 19},
        },
        range=[0, max(0.38, float(rates.max()) + 0.06)],
        dtick=0.05,
        tickformat=".2f",
        showgrid=True,
        gridcolor="#E6E6E6",
        gridwidth=1,
        zeroline=False,
        tickfont={"family": "Arial", "size": 16},
        fixedrange=True,
        row=1,
        col=3,
    )
    fig.update_yaxes(**common_yaxis, row=1, col=3)

    return fig


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------


def export_figure(fig: go.Figure) -> None:
    """Export Figure 7 in vector and high-resolution raster formats."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT_FIGURES.mkdir(parents=True, exist_ok=True)

    svg_path = OUTPUT_DIR / f"{OUTPUT_STEM}.svg"
    pdf_path = OUTPUT_DIR / f"{OUTPUT_STEM}.pdf"
    png_path = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
    tiff_path = OUTPUT_DIR / f"{OUTPUT_STEM}.tiff"

    fig.write_image(svg_path, width=FIGURE_WIDTH, height=FIGURE_HEIGHT)
    fig.write_image(pdf_path, width=FIGURE_WIDTH, height=FIGURE_HEIGHT)
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

    shutil.copy2(svg_path, MANUSCRIPT_FIGURES / svg_path.name)
    shutil.copy2(pdf_path, MANUSCRIPT_FIGURES / pdf_path.name)

    print("\nFigure 7 exported successfully:")
    print(f"  SVG : {svg_path}")
    print(f"  PDF : {pdf_path}")
    print(f"  PNG : {png_path}")
    print(f"  TIFF: {tiff_path}")
    print(f"  Manuscript directory: {MANUSCRIPT_FIGURES}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    """Generate and export Figure 7."""

    print(f"Project root: {ROOT}")

    stability, change_points, stability_file, change_file = load_data()

    print(f"Stability-summary input: {stability_file}")
    print(f"Change-point-summary input: {change_file}")

    stability, change_points = prepare_data(stability, change_points)

    print("\nStability ranking:")
    print(
        stability[
            [
                "company",
                "stability_index",
                "stability_rank",
                "stability_category",
            ]
        ].to_string(index=False)
    )

    print("\nNormalised instability components:")
    print(
        stability[
            ["company"] + [column for column, _ in INSTABILITY_COMPONENTS]
        ].to_string(index=False)
    )

    print("\nStructural change summary:")
    print(
        change_points[
            [
                "company",
                "n_change_points",
                "change_point_rate",
                "largest_detected_js",
                "largest_detected_transition",
                "largest_detected_severity",
            ]
        ].to_string(index=False)
    )

    fig = build_figure(stability, change_points)
    export_figure(fig)
    fig.show()


if __name__ == "__main__":
    main()
