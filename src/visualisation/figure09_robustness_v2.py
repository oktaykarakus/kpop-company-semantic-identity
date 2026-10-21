"""
Figure 9 - Robustness and Sensitivity Analysis

Panel A:
    Consolidated robustness conclusions across the six WP6 analyses.
    Columns distinguish overall robustness, continuous semantic
    structure, and exact ordinal ranking.

Panel B:
    Temporal-window sensitivity across annual, rolling 3-year,
    rolling 5-year, and rolling 7-year definitions.

Panel C:
    Selected quantitative robustness anchors reported directly from
    the frozen WP6 outputs.

Inputs:
    outputs/robustness/summary_robustness/
        wp6_primary_analysis_summary.csv
        wp6_global_robustness_summary.csv

    outputs/robustness/temporal_window_sensitivity/
        temporal_window_publication_summary.csv
        window_sensitivity.csv

    outputs/robustness/community_detection_sensitivity/
        community_detection_global_summary.csv

    outputs/robustness/metric_sensitivity/
        metric_sensitivity_summary.csv

Outputs:
    outputs/figures/fig09/
        fig09_robustness_sensitivity_v1.svg
        fig09_robustness_sensitivity_v1.pdf
        fig09_robustness_sensitivity_v1.png
        fig09_robustness_sensitivity_v1.tiff

    paper/manuscript/figures/
        fig09_robustness_sensitivity_v1.svg
        fig09_robustness_sensitivity_v1.pdf
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
    FIGURES,
    MANUSCRIPT_FIGURES,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ROBUSTNESS_DIR = ROOT / "outputs" / "robustness"

SUMMARY_DIR = ROBUSTNESS_DIR / "summary_robustness"
TEMPORAL_DIR = ROBUSTNESS_DIR / "temporal_window_sensitivity"
COMMUNITY_DIR = ROBUSTNESS_DIR / "community_detection_sensitivity"
METRIC_DIR = ROBUSTNESS_DIR / "metric_sensitivity"

PRIMARY_SUMMARY_FILE = (
    SUMMARY_DIR / "wp6_primary_analysis_summary.csv"
)

GLOBAL_SUMMARY_FILE = (
    SUMMARY_DIR / "wp6_global_robustness_summary.csv"
)

TEMPORAL_PUBLICATION_FILE = (
    TEMPORAL_DIR / "temporal_window_publication_summary.csv"
)

WINDOW_SENSITIVITY_FILE = (
    TEMPORAL_DIR / "window_sensitivity.csv"
)

COMMUNITY_GLOBAL_FILE = (
    COMMUNITY_DIR / "community_detection_global_summary.csv"
)

METRIC_SUMMARY_FILE = (
    METRIC_DIR / "metric_sensitivity_summary.csv"
)

OUTPUT_DIR = FIGURES / "fig09"
OUTPUT_STEM = "fig09_robustness_sensitivity_v2"

ANALYSIS_ORDER = [
    "06a",
    "06b",
    "06c",
    "06d",
    "06e",
    "06f",
]

ANALYSIS_LABELS = {
    "06a": "Similarity metric",
    "06b": "Bootstrap profiles",
    "06c": "Leave-one-artist-out",
    "06d": "Balanced sampling",
    "06e": "Temporal window",
    "06f": "Community detection",
}

CONCLUSION_COLUMNS = [
    ("robustness_conclusion", "Overall"),
    ("structure_conclusion", "Continuous structure"),
    ("ranking_conclusion", "Exact ranking"),
]

CONCLUSION_LEVELS = {
    "metric_sensitive": 1,
    "sensitive": 1,
    "moderately_robust": 2,
    "robust": 3,
    "highly_robust": 4,
}

CONCLUSION_LABELS = {
    1: "Sensitive",
    2: "Moderately robust",
    3: "Robust",
    4: "Highly robust",
}

ROBUSTNESS_COLOURSCALE = [
    [0.0000, "#D9D9D9"],
    [0.1666, "#D9D9D9"],
    [0.1667, "#F3D6A5"],
    [0.4999, "#F3D6A5"],
    [0.5000, "#A9D6C8"],
    [0.8332, "#A9D6C8"],
    [0.8333, "#3E8E7E"],
    [1.0000, "#3E8E7E"],
]

WINDOW_ORDER = [
    "annual",
    "rolling_3_year",
    "rolling_5_year",
    "rolling_7_year",
]

WINDOW_LABELS = {
    "annual": "Annual",
    "rolling_3_year": "3-year",
    "rolling_5_year": "5-year",
    "rolling_7_year": "7-year",
}

FIGURE_WIDTH = 1320
FIGURE_HEIGHT = 850
EXPORT_SCALE = 4


# ---------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------

def require_file(path: Path) -> None:
    """Raise a clear error when a required frozen output is absent."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required Figure 9 input was not found:\n{path}"
        )


def require_columns(
    frame: pd.DataFrame,
    required: set[str],
    name: str,
) -> None:
    """Validate required columns for one input table."""

    missing = sorted(required.difference(frame.columns))

    if missing:
        raise ValueError(
            f"{name} is missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )


def load_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load and validate the frozen WP6 outputs."""

    required_files = [
        PRIMARY_SUMMARY_FILE,
        GLOBAL_SUMMARY_FILE,
        TEMPORAL_PUBLICATION_FILE,
        WINDOW_SENSITIVITY_FILE,
        COMMUNITY_GLOBAL_FILE,
        METRIC_SUMMARY_FILE,
    ]

    for path in required_files:
        require_file(path)

    primary = pd.read_csv(PRIMARY_SUMMARY_FILE)
    global_summary = pd.read_csv(GLOBAL_SUMMARY_FILE)
    temporal_publication = pd.read_csv(
        TEMPORAL_PUBLICATION_FILE
    )
    window_sensitivity = pd.read_csv(
        WINDOW_SENSITIVITY_FILE
    )
    community_global = pd.read_csv(
        COMMUNITY_GLOBAL_FILE
    )
    metric_summary = pd.read_csv(
        METRIC_SUMMARY_FILE
    )

    require_columns(
        primary,
        {
            "analysis_code",
            "analysis_name",
            "robustness_conclusion",
            "structure_conclusion",
            "ranking_conclusion",
            "primary_metric",
            "primary_value",
            "secondary_metric",
            "secondary_value",
            "interpretation",
        },
        "wp6_primary_analysis_summary.csv",
    )

    require_columns(
        global_summary,
        {
            "n_expected_analyses",
            "n_available_analyses",
            "analysis_coverage",
            "overall_conclusion",
            "continuous_structure_conclusion",
            "exact_ranking_conclusion",
            "interpretation",
        },
        "wp6_global_robustness_summary.csv",
    )

    require_columns(
        temporal_publication,
        {
            "analysis",
            "entity",
            "primary_statistic",
            "primary_value",
            "secondary_statistic",
            "secondary_value",
            "agreement",
            "conclusion",
        },
        "temporal_window_publication_summary.csv",
    )

    require_columns(
        window_sensitivity,
        {
            "window_name",
            "window_width",
            "mean_consecutive_profile_drift",
            "dominant_community_change_probability",
            "converging_pair_fraction",
        },
        "window_sensitivity.csv",
    )

    require_columns(
        community_global,
        {
            "median_pairwise_distance_correlation",
            "exact_rank_order_probability",
            "median_distinctiveness_rank_correlation",
            "maximum_rank_displacement",
        },
        "community_detection_global_summary.csv",
    )

    require_columns(
        metric_summary,
        {
            "analysis",
            "baseline_metric",
            "alternative_metric",
            "spearman_r",
            "conclusion",
        },
        "metric_sensitivity_summary.csv",
    )

    observed_codes = set(primary["analysis_code"])
    expected_codes = set(ANALYSIS_ORDER)

    if observed_codes != expected_codes:
        raise ValueError(
            "WP6 primary summary does not contain the expected "
            "six analysis codes.\n"
            f"Expected: {ANALYSIS_ORDER}\n"
            f"Found: {sorted(observed_codes)}"
        )

    if len(global_summary) != 1:
        raise ValueError(
            "Expected exactly one global WP6 robustness row, "
            f"but found {len(global_summary)}."
        )

    if len(community_global) != 1:
        raise ValueError(
            "Expected exactly one community-detection global row, "
            f"but found {len(community_global)}."
        )

    return (
        primary,
        global_summary,
        temporal_publication,
        window_sensitivity,
        community_global,
        metric_summary,
    )


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------

def normalise_conclusion(value: str) -> str:
    """Normalise one categorical robustness conclusion."""

    value = str(value).strip().lower()

    if value not in CONCLUSION_LEVELS:
        raise ValueError(
            "Unknown robustness conclusion encountered: "
            f"{value!r}"
        )

    return value


def prepare_robustness_matrix(
    primary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare the ordered categorical robustness matrix."""

    ordered = (
        primary.set_index("analysis_code")
        .loc[ANALYSIS_ORDER]
        .reset_index()
    )

    numeric = pd.DataFrame(
        index=[
            ANALYSIS_LABELS[code]
            for code in ANALYSIS_ORDER
        ]
    )

    text = pd.DataFrame(index=numeric.index)

    for column, display_name in CONCLUSION_COLUMNS:
        conclusions = [
            normalise_conclusion(value)
            for value in ordered[column]
        ]

        numeric[display_name] = [
            CONCLUSION_LEVELS[value]
            for value in conclusions
        ]

        text[display_name] = [
            CONCLUSION_LABELS[
                CONCLUSION_LEVELS[value]
            ]
            for value in conclusions
        ]

    return numeric, text


def prepare_temporal_window_data(
    window_sensitivity: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare the ordered temporal-window sensitivity summary."""

    temporal = (
        window_sensitivity.set_index("window_name")
        .loc[WINDOW_ORDER]
        .reset_index()
    )

    for column in [
        "mean_consecutive_profile_drift",
        "dominant_community_change_probability",
        "converging_pair_fraction",
    ]:
        temporal[column] = pd.to_numeric(
            temporal[column],
            errors="coerce",
        )

        if temporal[column].isna().any():
            raise ValueError(
                f"Missing temporal-window values found in {column}."
            )

    temporal["display_window"] = temporal["window_name"].map(
        WINDOW_LABELS
    )

    return temporal


def find_metric_row(
    metric_summary: pd.DataFrame,
    analysis: str,
    alternative_metric: str,
) -> pd.Series:
    """Return one uniquely identified metric-sensitivity row."""

    matches = metric_summary[
        metric_summary["analysis"].eq(analysis)
        & metric_summary["alternative_metric"].eq(
            alternative_metric
        )
    ]

    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one metric-sensitivity row for "
            f"{analysis!r} and {alternative_metric!r}, "
            f"but found {len(matches)}."
        )

    return matches.iloc[0]


def find_primary_row(
    primary: pd.DataFrame,
    analysis_code: str,
) -> pd.Series:
    """Return one uniquely identified consolidated analysis row."""

    matches = primary[
        primary["analysis_code"].eq(analysis_code)
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected one row for analysis {analysis_code}, "
            f"but found {len(matches)}."
        )

    return matches.iloc[0]


def prepare_evidence_cards(
    primary: pd.DataFrame,
    temporal_publication: pd.DataFrame,
    community_global: pd.DataFrame,
    metric_summary: pd.DataFrame,
) -> list[dict[str, str]]:
    """Prepare directly validated quantitative evidence cards."""

    euclidean_pairwise = find_metric_row(
        metric_summary=metric_summary,
        analysis="Pairwise fingerprint distances",
        alternative_metric="euclidean",
    )

    bootstrap = find_primary_row(
        primary=primary,
        analysis_code="06b",
    )

    leave_one_out = find_primary_row(
        primary=primary,
        analysis_code="06c",
    )

    pairwise_temporal = temporal_publication[
        temporal_publication["analysis"].eq(
            "Pairwise temporal sensitivity"
        )
    ]

    if len(pairwise_temporal) != 6:
        raise ValueError(
            "Expected six pairwise temporal-sensitivity rows, "
            f"but found {len(pairwise_temporal)}."
        )

    full_pairwise_agreement = int(
        pairwise_temporal["primary_value"].eq(1.0).sum()
    )

    weakest_pair = pairwise_temporal.sort_values(
        "primary_value",
        ascending=True,
    ).iloc[0]

    community = community_global.iloc[0]

    cards = [
        {
            "value": f"{euclidean_pairwise['spearman_r']:.3f}",
            "title": "Cosine–Euclidean geometry",
            "detail": "Spearman correlation",
        },
        {
            "value": (
                f"{int(round(leave_one_out['primary_value'] * 126))}"
                "/126"
            ),
            "title": "Leave-one-artist-out",
            "detail": "complete ordering retained",
        },
        {
            "value": f"{bootstrap['secondary_value']:.2f}×",
            "title": "Bootstrap enrichment",
            "detail": "relative to random ordering",
        },
        {
            "value": f"{full_pairwise_agreement}/6",
            "title": "Temporal convergence",
            "detail": (
                "5 of 6 pairs fully retained; "
                f"weakest {weakest_pair['entity'].replace('|', '–')} "
                f"={weakest_pair['primary_value']:.2f}"
            ),
        },
        {
            "value": (
                f"{community['median_pairwise_distance_correlation']:.3f}"
            ),
            "title": "Community geometry",
            "detail": "median distance correlation",
        },
        {
            "value": (
                f"{community['exact_rank_order_probability']:.3f}"
            ),
            "title": "Exact rank-order retention",
            "detail": "retention probability",
        },
    ]

    return cards


# ---------------------------------------------------------------------
# Figure construction
# ---------------------------------------------------------------------

def build_figure(
    robustness_numeric: pd.DataFrame,
    robustness_text: pd.DataFrame,
    temporal: pd.DataFrame,
    evidence_cards: list[dict[str, str]],
    global_summary: pd.DataFrame,
) -> go.Figure:
    """Construct the final three-panel Figure 9."""

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [
                {"type": "heatmap"},
                {"type": "xy"},
            ],
            [
                {
                    "type": "xy",
                    "colspan": 2,
                },
                None,
            ],
        ],
        row_heights=[0.64, 0.36],
        column_widths=[0.53, 0.47],
        horizontal_spacing=0.12,
        vertical_spacing=0.21,
    )

    # Panel A: categorical robustness synthesis
    fig.add_trace(
        go.Heatmap(
            z=robustness_numeric.to_numpy(dtype=float),
            x=robustness_numeric.columns.tolist(),
            y=robustness_numeric.index.tolist(),
            text=robustness_text.to_numpy(),
            texttemplate="%{text}",
            textfont={
                "family": "Arial",
                "size": 15,
                "color": "black",
            },
            colorscale=ROBUSTNESS_COLOURSCALE,
            zmin=1,
            zmax=4,
            xgap=1.5,
            ygap=1.5,
            showscale=False,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Evidence type: %{x}<br>"
                "Conclusion: %{text}"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    # Panel B: temporal-window sensitivity
    fig.add_trace(
        go.Scatter(
            x=temporal["display_window"],
            y=temporal["mean_consecutive_profile_drift"],
            mode="lines+markers",
            name="Mean consecutive drift",
            line={
                "width": 3,
                "color": "#3B6FB6",
            },
            marker={
                "size": 11,
                "color": "#3B6FB6",
            },
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Mean consecutive drift: %{y:.3f}"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=temporal["display_window"],
            y=temporal[
                "dominant_community_change_probability"
            ],
            mode="lines+markers",
            name="Dominant-community change",
            line={
                "width": 3,
                "dash": "dash",
                "color": "#B6465F",
            },
            marker={
                "size": 11,
                "symbol": "diamond",
                "color": "#B6465F",
            },
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Dominant-community change: %{y:.3f}"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )

    # Panel C: evidence cards
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="markers",
            marker={
                "opacity": 0,
            },
            hoverinfo="skip",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # Arrange the six quantitative anchors as a tidy 3-column by
    # 2-row card grid. Every annotation uses the same fixed width
    # and height so that the evidence blocks align consistently.
    card_positions = [
        (0.17, 0.94),
        (0.50, 0.94),
        (0.83, 0.94),
        (0.17, 0.47),
        (0.50, 0.47),
        (0.83, 0.47),
    ]

    for (x_position, y_position), card in zip(
        card_positions,
        evidence_cards,
    ):
        fig.add_annotation(
            x=x_position,
            y=y_position,
            xref="x3 domain",
            yref="y3 domain",
            text=(
                f"<span style='font-size:24px'><b>"
                f"{card['value']}</b></span>"
                f"<br><span style='font-size:13px'><b>"
                f"{card['title']}</b></span>"
                f"<br><span style='font-size:11px'>"
                f"{card['detail']}</span>"
            ),
            showarrow=False,
            align="center",
            xanchor="center",
            yanchor="middle",
            width=255,
            height=92,
            bordercolor="#D6D6D6",
            borderwidth=1,
            borderpad=8,
            bgcolor="#FAFAFA",
            font={
                "family": "Arial",
                "size": 14,
                "color": "#222222",
            },
        )

    global_row = global_summary.iloc[0]

    # fig.add_annotation(
    #     x=0.5,
    #     y=-0.08,
    #     xref="x3 domain",
    #     yref="y3 domain",
    #     text=(
    #         "<b>Overall WP6 conclusion:</b> "
    #         f"{str(global_row['overall_conclusion']).replace('_', ' ')}. "
    #         "Continuous company-profile structure is more stable "
    #         "than exact ordinal ranking."
    #     ),
    #     showarrow=False,
    #     align="center",
    #     xanchor="center",
    #     yanchor="top",
    #     font={
    #         "family": "Arial",
    #         "size": 13,
    #         "color": "#333333",
    #     },
    # )

    # Panel labels
    panel_labels = [
        ("<b>(A)</b>", -0.055, 1.055),
        ("<b>(B)</b>", 0.565, 1.055),
        ("<b>(C)</b>", -0.055, 0.335),
    ]

    for text, x, y in panel_labels:
        fig.add_annotation(
            x=x,
            y=y,
            xref="paper",
            yref="paper",
            text=text,
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font={
                "family": "Arial",
                "size": 21,
                "color": "black",
            },
        )

    # Axes
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
        col=1,
    )

    fig.update_yaxes(
        title=None,
        autorange="reversed",
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
        col=2,
    )

    fig.update_yaxes(
        title={
            "text": "Sensitivity measure",
            "font": {
                "family": "Arial",
                "size": 15,
            },
        },
        range=[0, 0.75],
        tickformat=".1f",
        tickfont={
            "family": "Arial",
            "size": 14,
        },
        showgrid=True,
        gridcolor="#E8E8E8",
        gridwidth=1,
        zeroline=False,
        fixedrange=True,
        row=1,
        col=2,
    )

    fig.update_xaxes(
        visible=False,
        fixedrange=True,
        row=2,
        col=1,
    )

    fig.update_yaxes(
        visible=False,
        fixedrange=True,
        row=2,
        col=1,
    )

    fig.update_layout(
        width=FIGURE_WIDTH,
        height=FIGURE_HEIGHT,
        template="simple_white",
        margin={
            "l": 150,
            "r": 35,
            "t": 50,
            "b": 35,
        },
        font={
            "family": "Arial",
            "size": 14,
        },
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend={
            "orientation": "h",
            "x": 0.79,
            "xanchor": "center",
            "y": 1.085,
            "yanchor": "bottom",
            "font": {
                "family": "Arial",
                "size": 14,
            },
            "title": None,
        },
    )

    return fig


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------

def export_figure(fig: go.Figure) -> None:
    """Export Figure 9 in vector and high-resolution raster formats."""

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

    print("\nFigure 9 exported successfully:")
    print(f"  SVG : {svg_path}")
    print(f"  PDF : {pdf_path}")
    print(f"  PNG : {png_path}")
    print(f"  TIFF: {tiff_path}")
    print(f"  Manuscript directory: {MANUSCRIPT_FIGURES}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    """Generate and export Figure 9."""

    print(f"Project root: {ROOT}")
    print(f"Primary WP6 summary: {PRIMARY_SUMMARY_FILE}")
    print(f"Temporal-window summary: {WINDOW_SENSITIVITY_FILE}")
    print(f"Community summary: {COMMUNITY_GLOBAL_FILE}")
    print(f"Metric summary: {METRIC_SUMMARY_FILE}")

    (
        primary,
        global_summary,
        temporal_publication,
        window_sensitivity,
        community_global,
        metric_summary,
    ) = load_data()

    robustness_numeric, robustness_text = (
        prepare_robustness_matrix(primary)
    )

    temporal = prepare_temporal_window_data(
        window_sensitivity
    )

    evidence_cards = prepare_evidence_cards(
        primary=primary,
        temporal_publication=temporal_publication,
        community_global=community_global,
        metric_summary=metric_summary,
    )

    print("\nRobustness conclusions:")
    print(robustness_text.to_string())

    print("\nTemporal-window sensitivity:")
    print(
        temporal[
            [
                "display_window",
                "mean_consecutive_profile_drift",
                "dominant_community_change_probability",
                "converging_pair_fraction",
            ]
        ].to_string(index=False)
    )

    print("\nQuantitative robustness anchors:")
    for card in evidence_cards:
        print(
            f"  {card['title']}: "
            f"{card['value']} ({card['detail']})"
        )

    fig = build_figure(
        robustness_numeric=robustness_numeric,
        robustness_text=robustness_text,
        temporal=temporal,
        evidence_cards=evidence_cards,
        global_summary=global_summary,
    )

    export_figure(fig)
    fig.show()


if __name__ == "__main__":
    main()
