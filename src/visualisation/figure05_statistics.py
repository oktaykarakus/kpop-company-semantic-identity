"""
Figure 5 - Statistical Evidence for Company Differentiation

Panel A:
    Global Kruskal-Wallis tests across the Big Four companies.
    Bars show epsilon-squared effect sizes. Labels report the
    Kruskal-Wallis H statistic and Holm-adjusted p-value. Stars mark
    metrics significant after Holm correction at alpha = 0.05.

Panel B:
    Pairwise Mann-Whitney evidence, shown as -log10 of the
    Holm-adjusted p-value. Stars mark significant comparisons.

Panel C:
    Pairwise rank-biserial effect sizes. Positive values indicate
    larger artist-level values for the first company named in the
    column; negative values indicate larger values for the second.

Inputs:
    outputs/company_significance/tables/
        artist_level_global_metric_tests.csv
        artist_level_pairwise_metric_tests.csv

Outputs:
    outputs/figures/fig05/
        fig05_statistical_differentiation.svg
        fig05_statistical_differentiation.pdf
        fig05_statistical_differentiation.png
        fig05_statistical_differentiation.tiff

    paper/manuscript/figures/
        fig05_statistical_differentiation.svg
        fig05_statistical_differentiation.pdf
"""

from itertools import combinations
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
    COMPANY_SIGNIFICANCE,
    FIGURES,
    MANUSCRIPT_FIGURES,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

GLOBAL_TESTS_FILE = (
    COMPANY_SIGNIFICANCE
    / "tables"
    / "artist_level_global_metric_tests.csv"
)

PAIRWISE_TESTS_FILE = (
    COMPANY_SIGNIFICANCE
    / "tables"
    / "artist_level_pairwise_metric_tests.csv"
)

OUTPUT_DIR = FIGURES / "fig05"
OUTPUT_STEM = "fig05_statistical_differentiation"

METRIC_ORDER = [
    "participation_coefficient",
    "community_entropy",
    "external_strength_fraction",
    "external_communities_reached",
    "centroid_novelty",
    "local_novelty",
    "centroid_margin",
]

METRIC_LABELS = {
    "participation_coefficient": "Participation coefficient",
    "community_entropy": "Community entropy",
    "external_strength_fraction": "External strength fraction",
    "external_communities_reached": "External communities reached",
    "centroid_novelty": "Centroid novelty",
    "local_novelty": "Local novelty",
    "centroid_margin": "Centroid margin",
}

GLOBAL_COLOUR = "#3B6FB6"
GLOBAL_NONSIGNIFICANT_COLOUR = "#B8C7D9"

P_VALUE_COLOURSCALE = [
    [0.00, "#F7F7F7"],
    [0.35, "#D5E5EE"],
    [0.70, "#6FA6BF"],
    [1.00, "#2C5D7C"],
]

EFFECT_COLOURSCALE = [
    [0.00, "#2C5D7C"],
    [0.25, "#83B4C9"],
    [0.50, "#F7F7F7"],
    [0.75, "#E59A78"],
    [1.00, "#9D3D3D"],
]

ALPHA = 0.05
P_VALUE_LIMIT = 2.0
EFFECT_LIMIT = 1.0

FIGURE_WIDTH = 1480
FIGURE_HEIGHT = 820
EXPORT_SCALE = 4


# ---------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the frozen Figure 5 statistical outputs."""

    required_files = [
        GLOBAL_TESTS_FILE,
        PAIRWISE_TESTS_FILE,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required Figure 5 input was not found:\n{path}"
            )

    global_tests = pd.read_csv(GLOBAL_TESTS_FILE)
    pairwise_tests = pd.read_csv(PAIRWISE_TESTS_FILE)

    required_columns = {
        "global_tests": {
            "metric",
            "n_artists",
            "n_groups",
            "kruskal_h",
            "degrees_of_freedom",
            "raw_p_value",
            "epsilon_squared",
            "holm_adjusted_p",
            "significant_holm_005",
        },
        "pairwise_tests": {
            "metric",
            "company_1",
            "company_2",
            "n_1",
            "n_2",
            "mean_1",
            "mean_2",
            "median_1",
            "median_2",
            "mean_difference_1_minus_2",
            "mann_whitney_u",
            "raw_p_value",
            "rank_biserial",
            "holm_adjusted_p",
            "significant_holm_005",
        },
    }

    frames = {
        "global_tests": global_tests,
        "pairwise_tests": pairwise_tests,
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

    validate_global_tests(global_tests)
    validate_pairwise_tests(pairwise_tests)

    return global_tests, pairwise_tests


def validate_global_tests(global_tests: pd.DataFrame) -> None:
    """Validate the global Kruskal-Wallis results."""

    duplicated_metrics = global_tests.loc[
        global_tests["metric"].duplicated(),
        "metric",
    ].tolist()

    if duplicated_metrics:
        raise ValueError(
            "Duplicate metrics found in global tests: "
            f"{duplicated_metrics}"
        )

    observed_metrics = set(global_tests["metric"])
    expected_metrics = set(METRIC_ORDER)

    if observed_metrics != expected_metrics:
        raise ValueError(
            "Metrics in the global output do not match METRIC_ORDER.\n"
            f"Missing: {sorted(expected_metrics - observed_metrics)}\n"
            f"Unexpected: {sorted(observed_metrics - expected_metrics)}"
        )

    if not global_tests["n_groups"].eq(len(COMPANY_ORDER)).all():
        raise ValueError(
            "Global tests do not all contain the expected number "
            f"of groups ({len(COMPANY_ORDER)})."
        )

    numeric_columns = [
        "kruskal_h",
        "epsilon_squared",
        "holm_adjusted_p",
    ]

    if global_tests[numeric_columns].isna().any().any():
        raise ValueError(
            "Missing global-test statistics were found in required "
            "plotting columns."
        )

    invalid_p = ~global_tests["holm_adjusted_p"].between(
        0,
        1,
        inclusive="both",
    )

    if invalid_p.any():
        raise ValueError(
            "Global Holm-adjusted p-values outside [0, 1] were found."
        )

    if (global_tests["epsilon_squared"] < 0).any():
        raise ValueError(
            "Negative epsilon-squared values were found."
        )


def validate_pairwise_tests(pairwise_tests: pd.DataFrame) -> None:
    """Validate the complete metric-by-company-pair test grid."""

    duplicated_rows = (
        pairwise_tests.groupby(
            ["metric", "company_1", "company_2"],
            dropna=False,
        )
        .size()
    )
    duplicated_rows = duplicated_rows[duplicated_rows > 1]

    if not duplicated_rows.empty:
        raise ValueError(
            "Duplicate metric/company-pair rows found:\n"
            f"{duplicated_rows}"
        )

    observed_metrics = set(pairwise_tests["metric"])
    expected_metrics = set(METRIC_ORDER)

    if observed_metrics != expected_metrics:
        raise ValueError(
            "Metrics in the pairwise output do not match METRIC_ORDER.\n"
            f"Missing: {sorted(expected_metrics - observed_metrics)}\n"
            f"Unexpected: {sorted(observed_metrics - expected_metrics)}"
        )

    unknown_companies = sorted(
        set(pairwise_tests["company_1"])
        .union(pairwise_tests["company_2"])
        .difference(COMPANY_ORDER)
    )

    if unknown_companies:
        raise ValueError(
            "Unexpected companies found in pairwise tests: "
            f"{unknown_companies}"
        )

    pair_order = list(combinations(COMPANY_ORDER, 2))
    expected_rows = {
        (metric, company_1, company_2)
        for metric in METRIC_ORDER
        for company_1, company_2 in pair_order
    }
    observed_rows = set(
        zip(
            pairwise_tests["metric"],
            pairwise_tests["company_1"],
            pairwise_tests["company_2"],
        )
    )

    missing_rows = sorted(expected_rows.difference(observed_rows))
    unexpected_rows = sorted(observed_rows.difference(expected_rows))

    if missing_rows or unexpected_rows:
        raise ValueError(
            "Incomplete or incorrectly oriented pairwise-test grid.\n"
            f"Missing rows: {missing_rows}\n"
            f"Unexpected rows: {unexpected_rows}"
        )

    numeric_columns = [
        "holm_adjusted_p",
        "rank_biserial",
    ]

    if pairwise_tests[numeric_columns].isna().any().any():
        raise ValueError(
            "Missing pairwise statistics were found in required "
            "plotting columns."
        )

    invalid_p = ~pairwise_tests["holm_adjusted_p"].between(
        0,
        1,
        inclusive="both",
    )

    if invalid_p.any():
        raise ValueError(
            "Pairwise Holm-adjusted p-values outside [0, 1] were found."
        )

    invalid_effect = ~pairwise_tests["rank_biserial"].between(
        -1,
        1,
        inclusive="both",
    )

    if invalid_effect.any():
        raise ValueError(
            "Rank-biserial effect sizes outside [-1, 1] were found."
        )


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------

def format_adjusted_p(value: float) -> str:
    """Format adjusted p-values consistently for annotations."""

    if value < 0.001:
        return "<0.001"

    return f"{value:.3f}"


def prepare_data(
    global_tests: pd.DataFrame,
    pairwise_tests: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Prepare ordered global results and pairwise matrices."""

    global_ordered = (
        global_tests.set_index("metric")
        .reindex(METRIC_ORDER)
        .reset_index()
    )

    global_ordered["display_label"] = global_ordered["metric"].map(
        METRIC_LABELS
    )
    global_ordered["annotation"] = global_ordered.apply(
        lambda row: (
            f"H={row['kruskal_h']:.2f}<br>"
            f"p<sub>Holm</sub>={format_adjusted_p(row['holm_adjusted_p'])}"
            + ("   ★" if bool(row["significant_holm_005"]) else "")
        ),
        axis=1,
    )

    pair_order = list(combinations(COMPANY_ORDER, 2))
    pair_labels = [
        f"{company_1}–{company_2}"
        for company_1, company_2 in pair_order
    ]

    pairwise = pairwise_tests.copy()
    pairwise["pair_label"] = pairwise.apply(
        lambda row: f"{row['company_1']}–{row['company_2']}",
        axis=1,
    )

    adjusted_p_matrix = (
        pairwise.pivot(
            index="metric",
            columns="pair_label",
            values="holm_adjusted_p",
        )
        .reindex(index=METRIC_ORDER, columns=pair_labels)
    )

    significance_matrix = (
        pairwise.pivot(
            index="metric",
            columns="pair_label",
            values="significant_holm_005",
        )
        .reindex(index=METRIC_ORDER, columns=pair_labels)
        .astype(bool)
    )

    effect_matrix = (
        pairwise.pivot(
            index="metric",
            columns="pair_label",
            values="rank_biserial",
        )
        .reindex(index=METRIC_ORDER, columns=pair_labels)
    )

    return (
        global_ordered,
        adjusted_p_matrix,
        significance_matrix,
        effect_matrix,
    )


# ---------------------------------------------------------------------
# Hover data
# ---------------------------------------------------------------------

def create_pairwise_customdata(
    adjusted_p_matrix: pd.DataFrame,
    effect_matrix: pd.DataFrame,
) -> np.ndarray:
    """Create shared hover data for the two pairwise panels."""

    p_display = adjusted_p_matrix.copy().astype(object)
    effect_display = effect_matrix.copy().astype(object)

    for metric in adjusted_p_matrix.index:
        for pair in adjusted_p_matrix.columns:
            p_display.loc[metric, pair] = format_adjusted_p(
                float(adjusted_p_matrix.loc[metric, pair])
            )
            effect_display.loc[metric, pair] = (
                f"{float(effect_matrix.loc[metric, pair]):.3f}"
            )

    return np.dstack(
        [
            p_display.to_numpy(),
            effect_display.to_numpy(),
        ]
    )


# ---------------------------------------------------------------------
# Figure construction
# ---------------------------------------------------------------------

def build_figure(
    global_ordered: pd.DataFrame,
    adjusted_p_matrix: pd.DataFrame,
    significance_matrix: pd.DataFrame,
    effect_matrix: pd.DataFrame,
) -> go.Figure:
    """Construct the final three-panel Figure 5."""

    metric_labels = [METRIC_LABELS[metric] for metric in METRIC_ORDER]
    pair_labels = adjusted_p_matrix.columns.tolist()

    negative_log_p = -np.log10(
        adjusted_p_matrix.clip(lower=np.finfo(float).tiny)
    )
    plotting_negative_log_p = negative_log_p.clip(upper=P_VALUE_LIMIT)

    significance_text = np.where(
        significance_matrix.to_numpy(),
        "★",
        "",
    )

    pairwise_customdata = create_pairwise_customdata(
        adjusted_p_matrix=adjusted_p_matrix,
        effect_matrix=effect_matrix,
    )

    global_bar_colours = np.where(
        global_ordered["significant_holm_005"].astype(bool),
        GLOBAL_COLOUR,
        GLOBAL_NONSIGNIFICANT_COLOUR,
    )

    global_customdata = np.column_stack(
        [
            global_ordered["kruskal_h"],
            global_ordered["degrees_of_freedom"],
            global_ordered["holm_adjusted_p"],
            global_ordered["n_artists"],
        ]
    )

    fig = make_subplots(
        rows=1,
        cols=3,
        column_widths=[0.33, 0.34, 0.33],
        horizontal_spacing=0.075,
    )

    # Panel A: global Kruskal-Wallis tests
    fig.add_trace(
        go.Bar(
            x=global_ordered["epsilon_squared"],
            y=metric_labels,
            orientation="h",
            marker={
                "color": global_bar_colours,
                "line": {
                    "color": "white",
                    "width": 0.8,
                },
            },
            customdata=global_customdata,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "ε²: %{x:.3f}<br>"
                "Kruskal-Wallis H: %{customdata[0]:.3f}<br>"
                "Degrees of freedom: %{customdata[1]:.0f}<br>"
                "Holm-adjusted p: %{customdata[2]:.4f}<br>"
                "Artists: %{customdata[3]:.0f}"
                "<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    for row_index, row in global_ordered.iterrows():
        fig.add_annotation(
            x=float(row["epsilon_squared"]) + 0.004,
            y=metric_labels[row_index],
            xref="x",
            yref="y",
            text=row["annotation"],
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font={
                "family": "Arial",
                "size": 15,
                "color": "#222222",
            },
        )

    # Panel B: Holm-adjusted pairwise evidence
    fig.add_trace(
        go.Heatmap(
            z=plotting_negative_log_p.to_numpy(),
            x=pair_labels,
            y=metric_labels,
            text=significance_text,
            texttemplate="%{text}",
            textfont={
                "family": "Arial",
                "size": 21,
                "color": "black",
            },
            customdata=pairwise_customdata,
            colorscale=P_VALUE_COLOURSCALE,
            zmin=0,
            zmax=P_VALUE_LIMIT,
            colorbar={
                "title": {
                    "text": "−log₁₀ p<sub>Holm</sub>",
                    "side": "right",
                },
                "x": 0.655,
                "thickness": 12,
                "len": 0.66,
                "outlinewidth": 0,
                "tickfont": {
                    "family": "Arial",
                    "size": 14,
                },
            },
            xgap=1.3,
            ygap=1.3,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Pair: %{x}<br>"
                "Holm-adjusted p: %{customdata[0]}<br>"
                "Rank-biserial: %{customdata[1]}"
                "<extra></extra>"
            ),
            showscale=True,
        ),
        row=1,
        col=2,
    )

    # Panel C: rank-biserial effect sizes
    fig.add_trace(
        go.Heatmap(
            z=effect_matrix.to_numpy(),
            x=pair_labels,
            y=metric_labels,
            customdata=pairwise_customdata,
            colorscale=EFFECT_COLOURSCALE,
            zmin=-EFFECT_LIMIT,
            zmax=EFFECT_LIMIT,
            zmid=0,
            colorbar={
                "title": {
                    "text": "Rank-biserial r",
                    "side": "right",
                },
                "thickness": 12,
                "len": 0.66,
                "outlinewidth": 0,
                "tickfont": {
                    "family": "Arial",
                    "size": 14,
                },
            },
            xgap=1.3,
            ygap=1.3,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Pair: %{x}<br>"
                "Rank-biserial: %{customdata[1]}<br>"
                "Holm-adjusted p: %{customdata[0]}"
                "<extra></extra>"
            ),
            showscale=True,
        ),
        row=1,
        col=3,
    )

    fig.update_layout(
        width=FIGURE_WIDTH,
        height=FIGURE_HEIGHT,
        template="simple_white",
        margin={
            "l": 225,
            "r": 115,
            "t": 65,
            "b": 120,
        },
        font={
            "family": "Arial",
            "size": 16,
        },
        paper_bgcolor="white",
        plot_bgcolor="white",
        bargap=0.20,
    )

    # Panel labels
    panel_positions = [
        (-0.055, "<b>(A)</b>"),
        (0.355, "<b>(B)</b>"),
        (0.705, "<b>(C)</b>"),
    ]

    for x_position, label in panel_positions:
        fig.add_annotation(
            x=x_position,
            y=1.06,
            xref="paper",
            yref="paper",
            text=label,
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
            "text": "Epsilon-squared (ε²)",
            "font": {
                "family": "Arial",
                "size": 17,
            },
        },
        range=[0, 0.205],
        showgrid=True,
        gridcolor="#E6E6E6",
        gridwidth=1,
        zeroline=False,
        tickformat=".2f",
        tickfont={
            "family": "Arial",
            "size": 16,
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
            "size": 16,
        },
        showgrid=False,
        zeroline=False,
        fixedrange=True,
        row=1,
        col=1,
    )

    # Panels B and C axes
    for column in [2, 3]:
        fig.update_xaxes(
            title=None,
            side="top",
            tickangle=-42,
            tickfont={
                "family": "Arial",
                "size": 16,
            },
            showgrid=False,
            zeroline=False,
            fixedrange=True,
            row=1,
            col=column,
        )

        fig.update_yaxes(
            title=None,
            autorange="reversed",
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            fixedrange=True,
            row=1,
            col=column,
        )

    # Direction note for Panel C
    fig.add_annotation(
        x=0.835,
        y=-0.195,
        xref="paper",
        yref="paper",
        text=(
            "Positive: first company higher   |   "
            "Negative: second company higher"
        ),
        showarrow=False,
        xanchor="center",
        yanchor="top",
        font={
            "family": "Arial",
            "size": 15,
            "color": "#444444",
        },
    )

    return fig


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------

def export_figure(fig: go.Figure) -> None:
    """Export Figure 5 in vector and high-resolution raster formats."""

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

    print("\nFigure 5 exported successfully:")
    print(f"  SVG : {svg_path}")
    print(f"  PDF : {pdf_path}")
    print(f"  PNG : {png_path}")
    print(f"  TIFF: {tiff_path}")
    print(f"  Manuscript directory: {MANUSCRIPT_FIGURES}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    """Generate and export Figure 5."""

    print(f"Project root: {ROOT}")
    print(f"Global-test input: {GLOBAL_TESTS_FILE}")
    print(f"Pairwise-test input: {PAIRWISE_TESTS_FILE}")

    global_tests, pairwise_tests = load_data()

    (
        global_ordered,
        adjusted_p_matrix,
        significance_matrix,
        effect_matrix,
    ) = prepare_data(
        global_tests=global_tests,
        pairwise_tests=pairwise_tests,
    )

    print("\nGlobal tests:")
    print(
        global_ordered[
            [
                "metric",
                "kruskal_h",
                "epsilon_squared",
                "holm_adjusted_p",
                "significant_holm_005",
            ]
        ].to_string(index=False)
    )

    significant_pairwise = pairwise_tests.loc[
        pairwise_tests["significant_holm_005"].astype(bool),
        [
            "metric",
            "company_1",
            "company_2",
            "holm_adjusted_p",
            "rank_biserial",
        ],
    ]

    print("\nSignificant pairwise comparisons:")
    if significant_pairwise.empty:
        print("  None")
    else:
        print(significant_pairwise.to_string(index=False))

    fig = build_figure(
        global_ordered=global_ordered,
        adjusted_p_matrix=adjusted_p_matrix,
        significance_matrix=significance_matrix,
        effect_matrix=effect_matrix,
    )

    export_figure(fig)
    fig.show()


if __name__ == "__main__":
    main()
