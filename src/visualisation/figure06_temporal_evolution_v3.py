"""
Figure 6 - Temporal Evolution of Organisational Semantic Identity

Panel A:
    Grouped bars showing mean Jensen-Shannon semantic drift across early,
    middle, and late temporal phases for each Big Four entertainment company.

Panel B:
    Typical consecutive-period semantic drift. Points show the median
    Jensen-Shannon divergence and horizontal intervals show the
    interquartile range.

Panel C:
    Long-term temporal direction. Points show Theil-Sen slopes and
    horizontal intervals show their confidence bounds. The vertical
    zero line separates increasing from decreasing drift trajectories.

Inputs:
    outputs/temporal/rolling5/semantic_drift_summary_consecutive.csv
    outputs/temporal/rolling5/temporal_trajectory_summary.csv

Outputs:
    outputs/figures/fig06/
        fig06_temporal_evolution_v3.svg
        fig06_temporal_evolution_v3.pdf
        fig06_temporal_evolution_v3.png
        fig06_temporal_evolution_v3.tiff

    paper/manuscript/figures/
        fig06_temporal_evolution_v3.svg
        fig06_temporal_evolution_v3.pdf
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


TEMPORAL_ROLLING5 = (
    ROOT
    / "outputs"
    / "temporal"
    / "rolling5"
)

DRIFT_SUMMARY_FILE = (
    TEMPORAL_ROLLING5
    / "semantic_drift_summary_consecutive.csv"
)

TRAJECTORY_SUMMARY_FILE = (
    TEMPORAL_ROLLING5
    / "temporal_trajectory_summary.csv"
)

OUTPUT_DIR = FIGURES / "fig06"
OUTPUT_STEM = "fig06_temporal_evolution_v3"

COMPANY_COLOURS = {
    "HYBE": "#3B6FB6",
    "JYP": "#D9822B",
    "SM": "#2A9D8F",
    "YG": "#B6465F",
}

PHASE_COLUMNS = [
    "early_phase_mean",
    "middle_phase_mean",
    "late_phase_mean",
]

PHASE_LABELS = [
    "Early",
    "Middle",
    "Late",
]

FIGURE_WIDTH = 1500
FIGURE_HEIGHT = 720
EXPORT_SCALE = 4


# ---------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    """Load and validate the frozen Figure 6 temporal outputs."""

    required_files = [
        DRIFT_SUMMARY_FILE,
        TRAJECTORY_SUMMARY_FILE,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required Figure 6 input was not found:\n{path}"
            )

    drift = pd.read_csv(DRIFT_SUMMARY_FILE)
    trajectory = pd.read_csv(TRAJECTORY_SUMMARY_FILE)

    required_columns = {
        "drift": {
            "company",
            "n_transitions",
            "median_js",
            "q1_js",
            "q3_js",
        },
        "trajectory": {
            "company",
            "n_transitions",
            "early_phase_mean",
            "middle_phase_mean",
            "late_phase_mean",
            "theil_sen_slope",
            "theil_sen_low_slope",
            "theil_sen_high_slope",
            "kendall_tau",
            "kendall_p_value",
            "trend_classification",
        },
    }

    frames = {
        "drift": drift,
        "trajectory": trajectory,
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

    validate_company_table(
        frame=drift,
        name="drift",
    )

    validate_company_table(
        frame=trajectory,
        name="trajectory",
    )

    validate_cross_table_consistency(
        drift=drift,
        trajectory=trajectory,
    )

    validate_numeric_values(
        drift=drift,
        trajectory=trajectory,
    )

    return (
        drift,
        trajectory,
        DRIFT_SUMMARY_FILE,
        TRAJECTORY_SUMMARY_FILE,
    )


def validate_company_table(
    frame: pd.DataFrame,
    name: str,
) -> None:
    """Validate one-row-per-company structure and company coverage."""

    duplicated = frame.loc[
        frame["company"].duplicated(),
        "company",
    ].tolist()

    if duplicated:
        raise ValueError(
            f"Duplicate companies found in {name}: {duplicated}"
        )

    observed_companies = set(frame["company"])
    expected_companies = set(COMPANY_ORDER)

    if observed_companies != expected_companies:
        raise ValueError(
            f"Companies in {name} do not match COMPANY_ORDER.\n"
            f"Expected: {COMPANY_ORDER}\n"
            f"Found: {sorted(observed_companies)}"
        )


def validate_cross_table_consistency(
    drift: pd.DataFrame,
    trajectory: pd.DataFrame,
) -> None:
    """Check shared summary fields across the two frozen outputs."""

    merged = drift[
        ["company", "n_transitions"]
    ].merge(
        trajectory[["company", "n_transitions"]],
        on="company",
        suffixes=("_drift", "_trajectory"),
        validate="one_to_one",
    )

    inconsistent = merged.loc[
        merged["n_transitions_drift"]
        != merged["n_transitions_trajectory"]
    ]

    if not inconsistent.empty:
        raise ValueError(
            "Transition counts differ across Figure 6 inputs:\n"
            f"{inconsistent.to_string(index=False)}"
        )


def validate_numeric_values(
    drift: pd.DataFrame,
    trajectory: pd.DataFrame,
) -> None:
    """Validate ranges and interval ordering used in Figure 6."""

    drift_columns = [
        "median_js",
        "q1_js",
        "q3_js",
    ]

    trajectory_columns = PHASE_COLUMNS + [
        "theil_sen_slope",
        "theil_sen_low_slope",
        "theil_sen_high_slope",
        "kendall_tau",
        "kendall_p_value",
    ]

    for name, frame, columns in [
        ("drift", drift, drift_columns),
        ("trajectory", trajectory, trajectory_columns),
    ]:
        missing = frame[columns].isna()

        if missing.any().any():
            locations = np.argwhere(missing.to_numpy())
            details = [
                (
                    frame.iloc[row_index]["company"],
                    columns[column_index],
                )
                for row_index, column_index in locations
            ]

            raise ValueError(
                f"Missing numeric values found in {name}: {details}"
            )

    invalid_iqr = drift.loc[
        ~(
            (drift["q1_js"] <= drift["median_js"])
            & (drift["median_js"] <= drift["q3_js"])
        )
    ]

    if not invalid_iqr.empty:
        raise ValueError(
            "Invalid median/IQR ordering found:\n"
            f"{invalid_iqr.to_string(index=False)}"
        )

    invalid_slope_intervals = trajectory.loc[
        ~(
            (
                trajectory["theil_sen_low_slope"]
                <= trajectory["theil_sen_slope"]
            )
            & (
                trajectory["theil_sen_slope"]
                <= trajectory["theil_sen_high_slope"]
            )
        )
    ]

    if not invalid_slope_intervals.empty:
        raise ValueError(
            "Invalid Theil-Sen confidence intervals found:\n"
            f"{invalid_slope_intervals.to_string(index=False)}"
        )

    invalid_p_values = trajectory.loc[
        ~trajectory["kendall_p_value"].between(
            0,
            1,
            inclusive="both",
        )
    ]

    if not invalid_p_values.empty:
        raise ValueError(
            "Kendall p-values outside [0, 1] found:\n"
            f"{invalid_p_values.to_string(index=False)}"
        )


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------


def prepare_data(
    drift: pd.DataFrame,
    trajectory: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply deterministic company ordering to Figure 6 inputs."""

    ordered_drift = (
        drift.set_index("company")
        .reindex(COMPANY_ORDER)
        .reset_index()
    )

    ordered_trajectory = (
        trajectory.set_index("company")
        .reindex(COMPANY_ORDER)
        .reset_index()
    )

    return ordered_drift, ordered_trajectory


# ---------------------------------------------------------------------
# Figure construction
# ---------------------------------------------------------------------


def build_figure(
    drift: pd.DataFrame,
    trajectory: pd.DataFrame,
) -> go.Figure:
    """Construct the final three-panel Figure 6."""

    fig = make_subplots(
        rows=1,
        cols=3,
        column_widths=[0.40, 0.29, 0.31],
        horizontal_spacing=0.095,
    )

    # Panel A: grouped phase-specific temporal drift bars
    for company in COMPANY_ORDER:
        row = trajectory.loc[
            trajectory["company"].eq(company)
        ].iloc[0]

        phase_values = [
            float(row[column])
            for column in PHASE_COLUMNS
        ]

        fig.add_trace(
            go.Bar(
                x=PHASE_LABELS,
                y=phase_values,
                name=company,
                marker={
                    "color": COMPANY_COLOURS[company],
                    "line": {
                        "color": "white",
                        "width": 0.8,
                    },
                },
                customdata=np.array(
                    [[company, int(row["n_transitions"])]] * 3,
                    dtype=object,
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Phase: %{x}<br>"
                    "Mean JS drift: %{y:.4f}<br>"
                    "Transitions: %{customdata[1]}"
                    "<extra></extra>"
                ),
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    # Panel B: median drift with IQR
    median_values = drift["median_js"].to_numpy(dtype=float)
    q1_values = drift["q1_js"].to_numpy(dtype=float)
    q3_values = drift["q3_js"].to_numpy(dtype=float)

    fig.add_trace(
        go.Scatter(
            x=median_values,
            y=drift["company"],
            mode="markers",
            marker={
                "color": [
                    COMPANY_COLOURS[company]
                    for company in drift["company"]
                ],
                "size": 13,
                "line": {
                    "color": "white",
                    "width": 1.0,
                },
            },
            error_x={
                "type": "data",
                "symmetric": False,
                "array": q3_values - median_values,
                "arrayminus": median_values - q1_values,
                "thickness": 2.2,
                "width": 7,
                "color": "#505050",
            },
            customdata=np.column_stack(
                [
                    q1_values,
                    q3_values,
                    drift["n_transitions"],
                ]
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Median JS drift: %{x:.4f}<br>"
                "IQR: %{customdata[0]:.4f}--"
                "%{customdata[1]:.4f}<br>"
                "Transitions: %{customdata[2]:.0f}"
                "<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Panel C: Theil-Sen slope with interval
    slope_values = trajectory[
        "theil_sen_slope"
    ].to_numpy(dtype=float)

    low_values = trajectory[
        "theil_sen_low_slope"
    ].to_numpy(dtype=float)

    high_values = trajectory[
        "theil_sen_high_slope"
    ].to_numpy(dtype=float)

    fig.add_trace(
        go.Scatter(
            x=slope_values,
            y=trajectory["company"],
            mode="markers",
            marker={
                "color": [
                    COMPANY_COLOURS[company]
                    for company in trajectory["company"]
                ],
                "size": 13,
                "line": {
                    "color": "white",
                    "width": 1.0,
                },
            },
            error_x={
                "type": "data",
                "symmetric": False,
                "array": high_values - slope_values,
                "arrayminus": slope_values - low_values,
                "thickness": 2.2,
                "width": 7,
                "color": "#505050",
            },
            customdata=np.column_stack(
                [
                    low_values,
                    high_values,
                    trajectory["kendall_tau"],
                    trajectory["kendall_p_value"],
                    trajectory["trend_classification"],
                ]
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Theil-Sen slope: %{x:.5f}<br>"
                "Interval: %{customdata[0]:.5f}--"
                "%{customdata[1]:.5f}<br>"
                "Kendall tau: %{customdata[2]:.3f}<br>"
                "Kendall p: %{customdata[3]:.4f}<br>"
                "Trend: %{customdata[4]}"
                "<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=3,
    )

    fig.update_layout(
        width=FIGURE_WIDTH,
        height=FIGURE_HEIGHT,
        template="simple_white",
        barmode="group",
        bargap=0.18,
        bargroupgap=0.05,
        margin={
            "l": 95,
            "r": 55,
            "t": 75,
            "b": 105,
        },
        font={
            "family": "Arial",
            "size": 15,
        },
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend={
            "orientation": "h",
            "x": 0.20,
            "xanchor": "center",
            "y": -0.13,
            "yanchor": "top",
            "title": None,
            "font": {
                "family": "Arial",
                "size": 14,
            },
            "itemsizing": "constant",
        },
    )

    # Panel labels
    panel_positions = [
        (-0.02, "<b>(A)</b>"),
        (0.435, "<b>(B)</b>"),
        (0.735, "<b>(C)</b>"),
    ]

    for x_position, label in panel_positions:
        fig.add_annotation(
            x=x_position,
            y=1.065,
            xref="paper",
            yref="paper",
            text=label,
            showarrow=False,
            font={
                "family": "Arial",
                "size": 20,
            },
            xanchor="left",
            yanchor="top",
        )

    # Panel A axes
    fig.update_xaxes(
        title=None,
        tickfont={
            "family": "Arial",
            "size": 15,
        },
        showgrid=False,
        zeroline=False,
        fixedrange=True,
        row=1,
        col=1,
    )

    fig.update_yaxes(
        title={
            "text": "Mean consecutive Jensen-Shannon drift",
            "font": {
                "family": "Arial",
                "size": 17,
            },
        },
        rangemode="tozero",
        showgrid=True,
        gridcolor="#E6E6E6",
        gridwidth=1,
        zeroline=False,
        tickfont={
            "family": "Arial",
            "size": 14,
        },
        fixedrange=True,
        row=1,
        col=1,
    )

    # Panel B axes
    fig.update_xaxes(
        title={
            "text": "Median Jensen-Shannon drift",
            "font": {
                "family": "Arial",
                "size": 17,
            },
        },
        rangemode="tozero",
        showgrid=True,
        gridcolor="#E6E6E6",
        gridwidth=1,
        zeroline=False,
        tickfont={
            "family": "Arial",
            "size": 14,
        },
        fixedrange=True,
        row=1,
        col=2,
    )

    fig.update_yaxes(
        title=None,
        categoryorder="array",
        categoryarray=COMPANY_ORDER,
        autorange="reversed",
        showgrid=False,
        zeroline=False,
        tickfont={
            "family": "Arial",
            "size": 15,
        },
        fixedrange=True,
        row=1,
        col=2,
    )

    # Panel C axes
    fig.update_xaxes(
        title={
            "text": "Theil-Sen slope",
            "font": {
                "family": "Arial",
                "size": 17,
            },
        },
        showgrid=True,
        gridcolor="#E6E6E6",
        gridwidth=1,
        zeroline=True,
        zerolinecolor="#333333",
        zerolinewidth=1.5,
        tickfont={
            "family": "Arial",
            "size": 14,
        },
        tickformat=".3f",
        fixedrange=True,
        row=1,
        col=3,
    )

    fig.update_yaxes(
        title=None,
        categoryorder="array",
        categoryarray=COMPANY_ORDER,
        autorange="reversed",
        showgrid=False,
        zeroline=False,
        tickfont={
            "family": "Arial",
            "size": 15,
        },
        fixedrange=True,
        row=1,
        col=3,
    )

    fig.add_annotation(
        x=0.875,
        y=-0.115,
        xref="paper",
        yref="paper",
        text=(
            "Negative: declining drift over time"
            "  |  Positive: increasing drift over time"
        ),
        showarrow=False,
        font={
            "family": "Arial",
            "size": 13,
            "color": "#444444",
        },
        xanchor="center",
        yanchor="top",
    )

    return fig


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------


def export_figure(fig: go.Figure) -> None:
    """Export Figure 6 in vector and high-resolution raster formats."""

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

    print("\nFigure 6 exported successfully:")
    print(f"  SVG : {svg_path}")
    print(f"  PDF : {pdf_path}")
    print(f"  PNG : {png_path}")
    print(f"  TIFF: {tiff_path}")
    print(f"  Manuscript directory: {MANUSCRIPT_FIGURES}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    """Generate and export Figure 6."""

    print(f"Project root: {ROOT}")

    drift, trajectory, drift_file, trajectory_file = load_data()

    print(f"Drift-summary input: {drift_file}")
    print(f"Trajectory-summary input: {trajectory_file}")

    drift, trajectory = prepare_data(
        drift=drift,
        trajectory=trajectory,
    )

    print("\nPhase-specific mean JS drift:")
    print(
        trajectory[
            ["company"] + PHASE_COLUMNS
        ].to_string(index=False)
    )

    print("\nMedian JS drift and IQR:")
    print(
        drift[
            ["company", "median_js", "q1_js", "q3_js"]
        ].to_string(index=False)
    )

    print("\nTheil-Sen temporal slopes:")
    print(
        trajectory[
            [
                "company",
                "theil_sen_slope",
                "theil_sen_low_slope",
                "theil_sen_high_slope",
                "kendall_tau",
                "kendall_p_value",
                "trend_classification",
            ]
        ].to_string(index=False)
    )

    fig = build_figure(
        drift=drift,
        trajectory=trajectory,
    )

    export_figure(fig)
    fig.show()


if __name__ == "__main__":
    main()
