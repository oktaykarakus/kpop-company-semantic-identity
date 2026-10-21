"""
Figure 10 - Predictive Validation and Organisational Identifiability

Panel A:
    Predictive company classification using the combined semantic
    representation under song-level and artist-grouped validation.
    Macro F1 and balanced accuracy are shown for logistic regression,
    with 95% confidence intervals and the four-class chance level.

Panel B:
    Minimum catalogue fraction required to achieve 90%, 95%, and 99%
    identification accuracy for each company.

Panel C:
    Temporal persistence of company fingerprints under the persistence
    forecasting baseline, measured using community cosine similarity
    across 3-, 5-, and 7-year temporal windows.

Inputs:
    outputs/tables/prediction_results.csv
    outputs/tables/catalogue_recovery_curves.csv
    outputs/tables/forecast_results.csv

Outputs:
    outputs/figures/fig10/
        fig10_predictive_validation_v1.svg
        fig10_predictive_validation_v1.pdf
        fig10_predictive_validation_v1.png
        fig10_predictive_validation_v1.tiff

    paper/manuscript/figures/
        fig10_predictive_validation_v1.svg
        fig10_predictive_validation_v1.pdf
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
    FIGURES,
    MANUSCRIPT_FIGURES,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

TABLES_DIR = ROOT / "outputs" / "tables"

PREDICTION_FILE = TABLES_DIR / "prediction_results.csv"
CATALOGUE_FILE = TABLES_DIR / "catalogue_recovery_curves.csv"
FORECAST_FILE = TABLES_DIR / "forecast_results.csv"

OUTPUT_DIR = FIGURES / "fig10"
OUTPUT_STEM = "fig10_predictive_validation_v3"

COMPANY_COLOURS = {
    "HYBE": "#3B6FB6",
    "JYP": "#D9822B",
    "SM": "#2A9D8F",
    "YG": "#B6465F",
}

VALIDATION_ORDER = [
    "repeated_stratified_song_level",
    "repeated_stratified_artist_grouped",
]

VALIDATION_LABELS = {
    "repeated_stratified_song_level": "Song-level",
    "repeated_stratified_artist_grouped": "Artist-grouped",
}

WINDOW_ORDER = [3, 5, 7]

FIGURE_WIDTH = 1380
FIGURE_HEIGHT = 650
EXPORT_SCALE = 4


# ---------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------

def require_file(path: Path) -> None:
    """Raise a clear error when a required frozen output is absent."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required Figure 10 input was not found:\n{path}"
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
]:
    """Load and validate the frozen WP7 outputs."""

    for path in [
        PREDICTION_FILE,
        CATALOGUE_FILE,
        FORECAST_FILE,
    ]:
        require_file(path)

    prediction = pd.read_csv(PREDICTION_FILE)
    catalogue = pd.read_csv(CATALOGUE_FILE)
    forecast = pd.read_csv(FORECAST_FILE)

    require_columns(
        prediction,
        {
            "analysis",
            "feature_set",
            "validation_design",
            "model",
            "chance_accuracy",
            "balanced_accuracy_mean",
            "balanced_accuracy_ci95_low",
            "balanced_accuracy_ci95_high",
            "f1_macro_mean",
            "f1_macro_ci95_low",
            "f1_macro_ci95_high",
        },
        "prediction_results.csv",
    )

    require_columns(
        catalogue,
        {
            "company",
            "sample_fraction",
            "sample_size",
            "n_repeats",
            "identification_accuracy",
            "positive_margin_rate",
            "identification_margin_mean",
        },
        "catalogue_recovery_curves.csv",
    )

    require_columns(
        forecast,
        {
            "company",
            "window_size",
            "model",
            "n_forecasts",
            "community_cosine_similarity_mean",
            "community_cosine_similarity_ci_low",
            "community_cosine_similarity_ci_high",
            "community_cosine_similarity_mean",
            "interpretation",
        },
        "forecast_results.csv",
    )

    for name, frame in [
        ("catalogue", catalogue),
        ("forecast", forecast),
    ]:
        observed_companies = set(frame["company"])
        expected_companies = set(COMPANY_ORDER)

        if observed_companies != expected_companies:
            raise ValueError(
                f"Companies in {name} do not match COMPANY_ORDER.\n"
                f"Expected: {COMPANY_ORDER}\n"
                f"Found: {sorted(observed_companies)}"
            )

    return prediction, catalogue, forecast


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------

def prepare_prediction_data(
    prediction: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select the frozen main predictive comparison.

    The combined representation and logistic-regression classifier are
    used under both validation designs so that the comparison isolates
    the effect of validation structure rather than model or feature-set
    changes.
    """

    selected = prediction[
        prediction["analysis"].eq("Company prediction")
        & prediction["feature_set"].eq("combined")
        & prediction["model"].eq("logistic_regression")
        & prediction["validation_design"].isin(
            VALIDATION_ORDER
        )
    ].copy()

    if len(selected) != len(VALIDATION_ORDER):
        raise ValueError(
            "Expected one combined logistic-regression result for "
            "each validation design, but found:\n"
            f"{selected[['validation_design', 'model', 'feature_set']]}"
        )

    selected["validation_label"] = selected[
        "validation_design"
    ].map(VALIDATION_LABELS)

    selected["validation_order"] = selected[
        "validation_design"
    ].map(
        {
            design: index
            for index, design in enumerate(VALIDATION_ORDER)
        }
    )

    selected = selected.sort_values(
        "validation_order"
    ).reset_index(drop=True)

    return selected


def prepare_catalogue_data(
    catalogue: pd.DataFrame,
) -> pd.DataFrame:
    """Order and validate catalogue-recovery curves."""

    prepared = catalogue.copy()

    for column in [
        "sample_fraction",
        "identification_accuracy",
        "positive_margin_rate",
        "identification_margin_mean",
    ]:
        prepared[column] = pd.to_numeric(
            prepared[column],
            errors="coerce",
        )

        if prepared[column].isna().any():
            raise ValueError(
                f"Missing or non-numeric catalogue values in {column}."
            )

    if not prepared["sample_fraction"].between(
        0,
        1,
        inclusive="both",
    ).all():
        raise ValueError(
            "Catalogue fractions must lie within [0, 1]."
        )

    prepared["company_order"] = prepared["company"].map(
        {
            company: index
            for index, company in enumerate(COMPANY_ORDER)
        }
    )

    prepared = prepared.sort_values(
        ["company_order", "sample_fraction"]
    ).reset_index(drop=True)

    return prepared


def prepare_forecast_data(
    forecast: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select the persistence baseline across all companies and windows.

    A common baseline is retained across every company and temporal
    definition to avoid selecting different models post hoc.
    """

    prepared = forecast[
        forecast["model"].eq("persistence")
        & forecast["window_size"].isin(WINDOW_ORDER)
    ].copy()

    expected_rows = len(COMPANY_ORDER) * len(WINDOW_ORDER)

    if len(prepared) != expected_rows:
        raise ValueError(
            "Expected one persistence result for every company-window "
            f"combination ({expected_rows} rows), but found "
            f"{len(prepared)}."
        )

    duplicated = prepared.duplicated(
        ["company", "window_size"]
    )

    if duplicated.any():
        raise ValueError(
            "Duplicate persistence forecasting rows found:\n"
            f"{prepared.loc[duplicated, ['company', 'window_size']]}"
        )

    prepared["company_order"] = prepared["company"].map(
        {
            company: index
            for index, company in enumerate(COMPANY_ORDER)
        }
    )

    prepared = prepared.sort_values(
        ["company_order", "window_size"]
    ).reset_index(drop=True)

    return prepared



def compute_catalogue_thresholds(
    catalogue: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the minimum catalogue fraction reaching each target."""

    targets = [0.90, 0.95, 0.99]
    records: list[dict[str, float | str]] = []

    for company in COMPANY_ORDER:
        company_data = (
            catalogue[catalogue["company"].eq(company)]
            .sort_values("sample_fraction")
        )

        record: dict[str, float | str] = {
            "company": company,
        }

        for target_accuracy in targets:
            reached = company_data[
                company_data["identification_accuracy"]
                >= target_accuracy
            ]

            if reached.empty:
                fraction = np.nan
            else:
                fraction = float(
                    reached["sample_fraction"].iloc[0]
                )

            record[f"fraction_{int(target_accuracy * 100)}"] = (
                fraction
            )

        records.append(record)

    thresholds = pd.DataFrame(records)

    if thresholds[
        [
            "fraction_90",
            "fraction_95",
            "fraction_99",
        ]
    ].isna().any().any():
        raise ValueError(
            "At least one company did not reach a required catalogue "
            "identification target."
        )

    return thresholds


# ---------------------------------------------------------------------
# Figure construction
# ---------------------------------------------------------------------

def build_figure(
    prediction: pd.DataFrame,
    catalogue: pd.DataFrame,
    thresholds: pd.DataFrame,
    forecast: pd.DataFrame,
) -> go.Figure:
    """Construct the final three-panel Figure 10."""

    fig = make_subplots(
        rows=1,
        cols=3,
        column_widths=[0.30, 0.40, 0.30],
        horizontal_spacing=0.105,
    )

    # Panel A: predictive classification
    x_labels = prediction["validation_label"].tolist()

    for metric_name, mean_column, low_column, high_column, colour in [
        (
            "Balanced accuracy",
            "balanced_accuracy_mean",
            "balanced_accuracy_ci95_low",
            "balanced_accuracy_ci95_high",
            "#3B6FB6",
        ),
        (
            "Macro F1",
            "f1_macro_mean",
            "f1_macro_ci95_low",
            "f1_macro_ci95_high",
            "#D9822B",
        ),
    ]:
        means = prediction[mean_column].to_numpy(dtype=float)
        lows = prediction[low_column].to_numpy(dtype=float)
        highs = prediction[high_column].to_numpy(dtype=float)

        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=means,
                name=metric_name,
                legendgroup=f"panel_a_{metric_name}",
                legend="legend",
                marker={
                    "color": colour,
                    "line": {
                        "color": "white",
                        "width": 0.8,
                    },
                },
                error_y={
                    "type": "data",
                    "array": highs - means,
                    "arrayminus": means - lows,
                    "visible": True,
                    "thickness": 1.5,
                    "width": 4,
                    "color": "#333333",
                },
                text=[
                    f"{value:.3f}"
                    for value in means
                ],
                textposition="outside",
                textfont={
                    "family": "Arial",
                    "size": 12,
                },
                cliponaxis=False,
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    f"{metric_name}: %{{y:.3f}}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    chance_level = float(
        prediction["chance_accuracy"].iloc[0]
    )

    fig.add_hline(
        y=chance_level,
        line={
            "color": "#C00000",
            "width": 2.2,
            "dash": "dash",
        },
        annotation_text="<b>Chance = 0.25</b>",
        annotation_position="bottom right",
        annotation_font={
            "family": "Arial",
            "size": 12,
            "color": "#C00000",
        },
        row=1,
        col=1,
    )

    # Panel B: minimum catalogue fraction for target accuracy
    #
    # Target-accuracy levels form the x-axis groups. Within every
    # group, one bar is shown per company using the canonical company
    # colours. This avoids using transparency as an implicit legend.
    target_specs = [
        ("fraction_90", ">=90%"),
        ("fraction_95", ">=95%"),
        ("fraction_99", ">=99%"),
    ]

    target_labels = [
        label
        for _, label in target_specs
    ]

    for company in COMPANY_ORDER:
        company_row = thresholds[
            thresholds["company"].eq(company)
        ]

        if len(company_row) != 1:
            raise ValueError(
                f"Expected one catalogue-threshold row for {company}, "
                f"but found {len(company_row)}."
            )

        values = [
            float(company_row[column].iloc[0])
            for column, _ in target_specs
        ]

        fig.add_trace(
            go.Bar(
                x=target_labels,
                y=values,
                name=company,
                legendgroup=f"panel_b_{company}",
                legend="legend2",
                marker={
                    "color": COMPANY_COLOURS[company],
                    "line": {
                        "color": "white",
                        "width": 0.8,
                    },
                },
                text=[
                    f"{value:.0%}"
                    for value in values
                ],
                textposition="outside",
                textfont={
                    "family": "Arial",
                    "size": 11,
                },
                cliponaxis=False,
                hovertemplate=(
                    f"<b>{company}</b><br>"
                    "Target accuracy: %{x}<br>"
                    "Minimum catalogue fraction: %{y:.0%}"
                    "<extra></extra>"
                ),
                showlegend=True,
            ),
            row=1,
            col=2,
        )

    # Panel C: temporal persistence
    for company in COMPANY_ORDER:
        company_data = forecast[
            forecast["company"].eq(company)
        ]

        means = company_data[
            "community_cosine_similarity_mean"
        ].to_numpy(dtype=float)

        lows = company_data[
            "community_cosine_similarity_ci_low"
        ].to_numpy(dtype=float)

        highs = company_data[
            "community_cosine_similarity_ci_high"
        ].to_numpy(dtype=float)

        fig.add_trace(
            go.Scatter(
                x=company_data["window_size"],
                y=means,
                mode="lines+markers",
                name=company,
                legendgroup=f"panel_c_{company}",
                legend="legend3",
                showlegend=True,
                line={
                    "width": 2.8,
                    "color": COMPANY_COLOURS[company],
                },
                marker={
                    "size": 8,
                    "color": COMPANY_COLOURS[company],
                },
                error_y={
                    "type": "data",
                    "array": highs - means,
                    "arrayminus": means - lows,
                    "visible": True,
                    "thickness": 1.2,
                    "width": 3,
                    "color": COMPANY_COLOURS[company],
                },
                customdata=np.column_stack(
                    [
                        company_data["n_forecasts"],
                        company_data[
                            "community_cosine_similarity_mean"
                        ],
                    ]
                ),
                hovertemplate=(
                    f"<b>{company}</b><br>"
                    "Window: %{x:.0f} years<br>"
                    "Community cosine similarity: %{y:.3f}<br>"
                    "Forecasts: %{customdata[0]:.0f}<br>"
                    "Community cosine similarity: "
                    "%{customdata[1]:.3f}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=3,
        )

    # Panel labels
    for text, x in [
        ("<b>(A)</b>", -0.045),
        ("<b>(B)</b>", 0.305),
        ("<b>(C)</b>", 0.735),
    ]:
        fig.add_annotation(
            x=x,
            y=1.065,
            xref="paper",
            yref="paper",
            text=text,
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font={
                "family": "Arial",
                "size": 19,
                "color": "black",
            },
        )

    # Panel A axes
    fig.update_xaxes(
        title=None,
        tickfont={
            "family": "Arial",
            "size": 13,
        },
        showgrid=False,
        zeroline=False,
        fixedrange=True,
        row=1,
        col=1,
    )

    fig.update_yaxes(
        title={
            "text": "Predictive performance",
            "font": {
                "family": "Arial",
                "size": 15,
            },
        },
        range=[0, 0.52],
        tickformat=".1f",
        tickfont={
            "family": "Arial",
            "size": 12,
        },
        showgrid=True,
        gridcolor="#E8E8E8",
        zeroline=False,
        fixedrange=True,
        row=1,
        col=1,
    )

    # Panel B axes
    fig.update_xaxes(
        title={
            "text": "Target identification accuracy",
            "font": {
                "family": "Arial",
                "size": 15,
            },
        },
        tickfont={
            "family": "Arial",
            "size": 13,
        },
        showgrid=False,
        zeroline=False,
        fixedrange=True,
        row=1,
        col=2,
    )

    fig.update_yaxes(
        title={
            "text": "Minimum catalogue fraction",
            "font": {
                "family": "Arial",
                "size": 15,
            },
        },
        range=[0, 0.58],
        tickformat=".0%",
        dtick=0.1,
        tickfont={
            "family": "Arial",
            "size": 12,
        },
        showgrid=True,
        gridcolor="#E8E8E8",
        zeroline=False,
        fixedrange=True,
        row=1,
        col=2,
    )

    # Panel C axes
    fig.update_xaxes(
        title={
            "text": "Temporal window (years)",
            "font": {
                "family": "Arial",
                "size": 15,
            },
        },
        tickvals=WINDOW_ORDER,
        range=[2.5, 7.5],
        tickfont={
            "family": "Arial",
            "size": 12,
        },
        showgrid=False,
        zeroline=False,
        fixedrange=True,
        row=1,
        col=3,
    )

    fig.update_yaxes(
        title={
            "text": "Community cosine similarity",
            "font": {
                "family": "Arial",
                "size": 15,
            },
        },
        range=[0.80, 1.01],
        tickformat=".1f",
        tickfont={
            "family": "Arial",
            "size": 12,
        },
        showgrid=True,
        gridcolor="#E8E8E8",
        zeroline=False,
        fixedrange=True,
        row=1,
        col=3,
    )

    fig.update_layout(
        width=FIGURE_WIDTH,
        height=FIGURE_HEIGHT,
        template="simple_white",
        margin={
            "l": 80,
            "r": 45,
            "t": 55,
            "b": 55,
        },
        font={
            "family": "Arial",
            "size": 14,
        },
        paper_bgcolor="white",
        plot_bgcolor="white",
        barmode="group",
        bargap=0.25,
        bargroupgap=0.08,
        legend={
            "orientation": "h",
            "x": 0.135,
            "xanchor": "center",
            "y": 0.95,
            "yanchor": "bottom",
            "font": {
                "family": "Arial",
                "size": 11,
            },
            "title": None,
        },
        legend2={
            "orientation": "h",
            "x": 0.505,
            "xanchor": "center",
            "y": 0.95,
            "yanchor": "bottom",
            "font": {
                "family": "Arial",
                "size": 11,
            },
            "title": None,
        },
        legend3={
            "orientation": "h",
            "x": 0.9,
            "xanchor": "center",
            "y": 0.95,
            "yanchor": "bottom",
            "font": {
                "family": "Arial",
                "size": 11,
            },
            "title": None,
        },
    )

    return fig


# ---------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------

def export_figure(fig: go.Figure) -> None:
    """Export Figure 10 in vector and high-resolution raster formats."""

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

    print("\nFigure 10 exported successfully:")
    print(f"  SVG : {svg_path}")
    print(f"  PDF : {pdf_path}")
    print(f"  PNG : {png_path}")
    print(f"  TIFF: {tiff_path}")
    print(f"  Manuscript directory: {MANUSCRIPT_FIGURES}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    """Generate and export Figure 10."""

    print(f"Project root: {ROOT}")
    print(f"Prediction input: {PREDICTION_FILE}")
    print(f"Catalogue input: {CATALOGUE_FILE}")
    print(f"Forecast input: {FORECAST_FILE}")

    prediction_raw, catalogue_raw, forecast_raw = load_data()

    prediction = prepare_prediction_data(
        prediction_raw
    )

    catalogue = prepare_catalogue_data(
        catalogue_raw
    )

    thresholds = compute_catalogue_thresholds(
        catalogue
    )

    forecast = prepare_forecast_data(
        forecast_raw
    )

    print("\nSelected predictive classification results:")
    print(
        prediction[
            [
                "validation_label",
                "balanced_accuracy_mean",
                "balanced_accuracy_ci95_low",
                "balanced_accuracy_ci95_high",
                "f1_macro_mean",
                "f1_macro_ci95_low",
                "f1_macro_ci95_high",
                "chance_accuracy",
            ]
        ].to_string(index=False)
    )

    print("\nCatalogue recovery at 10%, 50%, and 90%:")
    print(
        catalogue[
            catalogue["sample_fraction"].isin(
                [0.1, 0.5, 0.9]
            )
        ][
            [
                "company",
                "sample_fraction",
                "sample_size",
                "identification_accuracy",
                "identification_margin_mean",
            ]
        ].to_string(index=False)
    )

    print("\nPersistence forecasting results:")
    print(
        forecast[
            [
                "company",
                "window_size",
                "community_cosine_similarity_mean",
                "community_cosine_similarity_ci_low",
                "community_cosine_similarity_ci_high",
                "community_cosine_similarity_mean",
            ]
        ].to_string(index=False)
    )

    fig = build_figure(
        prediction=prediction,
        catalogue=catalogue,
        thresholds=thresholds,
        forecast=forecast,
    )

    export_figure(fig)
    fig.show()


if __name__ == "__main__":
    main()
