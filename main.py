from dpsdc.etl import (
    begin_workshop,
    load_view,
    categorical_features,
    load_channels,
    load_configs,
)
from dpsdc.models import LogisticRegressionModel, RandomForestModel

from tableone import TableOne
import json

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


from dotenv import dotenv_values
from pathlib import Path
import argparse
from itertools import product

CONFIGS = load_configs()

STATS_FEATURES = (
    CONFIGS["OUTCOMES"]
    + CONFIGS["PROXY"]
    + CONFIGS["TRADITIONAL_LABELS"]
    + CONFIGS["VARIABLES"]
)
COVARIATES = load_channels(CONFIGS["COVARIATES_CHANNELS"])
ALL_FEATURES = STATS_FEATURES + COVARIATES

OUTPUT_PATH = Path("./output")
PROJECT_ID = dotenv_values(".env")["project_id"]


def discrete_multivariate_histogram(dataset: pd.DataFrame, x: str, hue: str):
    if dataset[x].dtypes == float:
        DISCRETE = False
    else:
        DISCRETE = True

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    _ = sns.histplot(
        dataset,
        x=x,
        hue=hue,
        discrete=DISCRETE,
        common_norm=False,
        stat="count",
        multiple="stack",
        ax=ax[0],
    )
    _ = sns.histplot(
        dataset,
        x=x,
        hue=hue,
        discrete=DISCRETE,
        common_norm=True,
        stat="probability",
        multiple="fill",
        ax=ax[1],
    )
    _ = sns.histplot(
        dataset,
        x=x,
        hue=hue,
        discrete=DISCRETE,
        common_norm=False,
        stat="proportion",
        multiple="fill",
        ax=ax[2],
    )

    fig.suptitle(f"{x} by {hue}")
    _ = ax[0].set_title("Count")
    _ = ax[1].set_title("Probability")
    _ = ax[2].set_title("Proportion")
    return fig


if __name__ == "__main__":
    begin_workshop(PROJECT_ID)
    print("1. Initialization")

    dataset = load_view(ALL_FEATURES, None)
    dataset = dataset[(dataset.spo2_mean > 85) & (dataset.spo2_mean.notna())]
    print("2. Data Loading")

    table = TableOne(
        dataset[STATS_FEATURES],
        categorical=categorical_features(STATS_FEATURES),
        pval=True,
        groupby=CONFIGS["PROXY"],
    )
    table.to_html(OUTPUT_PATH / "table_1.html")
    print("3. Table One Generated")

    train_set, test_set = train_test_split(dataset, test_size=CONFIGS["TEST_SIZE"])
    lr_abg = LogisticRegressionModel(outcome=CONFIGS["PROXY"]).fit(train_set)
    rf_abg = RandomForestModel(
        outcome=CONFIGS["PROXY"],
    ).fit(train_set)

    lr_outcome = LogisticRegressionModel(
        outcome=CONFIGS["OUTCOMES"],
    ).fit(train_set)
    rf_outcome = RandomForestModel(
        outcome=CONFIGS["OUTCOMES"],
    ).fit(train_set)
    print("4. Models Training Complete")

    with PdfPages(OUTPUT_PATH / "report.pdf") as pdf:
        fig, ax = plt.subplots(2, 4, figsize=(20, 10))
        for cax, (x, y, hue) in zip(
            ax.flatten(),
            product(
                CONFIGS["PROXY"], CONFIGS["VARIABLES"], CONFIGS["TRADITIONAL_LABELS"]
            ),
        ):
            _ = sns.boxplot(dataset, x=x, y=y, hue=hue, showfliers=False, ax=cax)
        pdf.savefig(fig)
        print("5. Boxplots Generation 1/2")

        fig, ax = plt.subplots(2, 4, figsize=(20, 10))
        for cax, (x, y, hue) in zip(
            ax.flatten(),
            product(
                CONFIGS["OUTCOMES"], CONFIGS["VARIABLES"], CONFIGS["TRADITIONAL_LABELS"]
            ),
        ):
            _ = sns.boxplot(dataset, x=x, y=y, hue=hue, showfliers=False, ax=cax)
        pdf.savefig(fig)
        print("6. Boxplots Generation 2/2")

        for x, hue in product(
            CONFIGS["VARIABLES"],
            CONFIGS["TRADITIONAL_LABELS"] + CONFIGS["PROXY"] + CONFIGS["OUTCOMES"],
        ):
            pdf.savefig(discrete_multivariate_histogram(dataset, x, hue))
        print("7. Histograms Generation")

        pdf.savefig(lr_abg.plot())
        pdf.savefig(lr_abg.beeswarm(train_set, test_set))
        pdf.savefig(lr_abg.plot_curves())
        print("8. Model Tracing 1/4")

        pdf.savefig(rf_abg.plot())
        pdf.savefig(rf_abg.beeswarm(train_set, test_set))
        pdf.savefig(rf_abg.plot_curves())
        print("9. Model Tracing 2/4")

        pdf.savefig(lr_outcome.plot())
        pdf.savefig(lr_outcome.beeswarm(train_set, test_set))
        pdf.savefig(lr_outcome.plot_curves())
        print("10. Model Tracing 3/4")

        pdf.savefig(rf_outcome.plot())
        pdf.savefig(rf_outcome.beeswarm(train_set, test_set))
        pdf.savefig(rf_outcome.plot_curves())
        print("11. Model Tracing 4/4")


print("12. Report Generation Completed")
