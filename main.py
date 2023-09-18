from dpsdc.etl import begin_workshop, load_view, categorical_features, load_channels
from dpsdc.models import LogisticRegressionModel, RandomForestModel

from tableone import TableOne

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

# Params
TRADITIONAL_LABELS = ["race", "gender", "language", "insurance"]
SEVERITY = ["SOFA", "charlson_comorbidity_index"]
DEMOGRAPHIC = ["admission_age", "careunit"]
PROXIES = ["received_abg_at_T"]
FEATURES = SEVERITY + TRADITIONAL_LABELS + PROXIES + DEMOGRAPHIC
OUTCOMES = ["hospital_death"]
STATS_FEATURES = FEATURES + OUTCOMES
RANDOM_STATE = 0
TEST_SIZE = 0.25
OUTPUT_PATH = Path("./output")
PROJECT_ID = dotenv_values(".env")["project_id"]


def discrete_multivariate_histogram(dataset: pd.DataFrame, x: str, hue: str):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    _ = sns.histplot(
        dataset,
        x=x,
        hue=hue,
        discrete=True,
        common_norm=False,
        stat="count",
        multiple="stack",
        ax=ax[0],
    )
    _ = sns.histplot(
        dataset,
        x=x,
        hue=hue,
        discrete=True,
        common_norm=True,
        stat="probability",
        multiple="fill",
        ax=ax[1],
    )
    _ = sns.histplot(
        dataset,
        x=x,
        hue=hue,
        discrete=True,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dry", action="store_true")
    args = parser.parse_args()
    if args.dry:
        N_REPEATS = 1
        print(f"DRY RUN - N_REPEATS set to {N_REPEATS}")
    else:
        N_REPEATS = 10
        print(f"LIVE RUN - N_REPEATS set to {N_REPEATS}")

    begin_workshop(PROJECT_ID)
    print("1. Initialization")

    COVARIATES = load_channels()["vitals"] + load_channels()["lab"]
    ALL_FEATURES = STATS_FEATURES + COVARIATES

    dataset = load_view(ALL_FEATURES, None)
    dataset = dataset[dataset.spo2_mean > 85]
    print("2. Data Loading")

    table = TableOne(
        dataset[STATS_FEATURES],
        categorical=categorical_features(STATS_FEATURES),
        pval=True,
        groupby=PROXIES,
    )
    table.to_html(OUTPUT_PATH / "table_1.html")
    print("3. Table One Generated")

    train_set, test_set = train_test_split(dataset, test_size=TEST_SIZE)
    lr_abg = LogisticRegressionModel(
        outcome="received_abg_at_T",
        features=FEATURES,
        covariates=COVARIATES,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    ).fit(train_set)
    rf_abg = RandomForestModel(
        outcome="received_abg_at_T",
        features=FEATURES,
        covariates=COVARIATES,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    ).fit(train_set)

    lr_outcome = LogisticRegressionModel(
        outcome="hospital_death",
        features=FEATURES,
        covariates=COVARIATES,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    ).fit(train_set)
    rf_outcome = RandomForestModel(
        outcome="hospital_death",
        features=FEATURES,
        covariates=COVARIATES,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    ).fit(train_set)
    print("4. Models Training Complete")

    with PdfPages(OUTPUT_PATH / "report.pdf") as pdf:
        fig, ax = plt.subplots(2, 4, figsize=(20, 10))
        for cax, (x, y, hue) in zip(
            ax.flatten(), product(PROXIES, SEVERITY, TRADITIONAL_LABELS)
        ):
            _ = sns.boxplot(dataset, x=x, y=y, hue=hue, showfliers=False, ax=cax)
        pdf.savefig(fig)
        print("5. Boxplots Generation 1/2")

        fig, ax = plt.subplots(2, 4, figsize=(20, 10))
        for cax, (x, y, hue) in zip(
            ax.flatten(), product(OUTCOMES, SEVERITY, TRADITIONAL_LABELS)
        ):
            _ = sns.boxplot(dataset, x=x, y=y, hue=hue, showfliers=False, ax=cax)
        pdf.savefig(fig)
        print("6. Boxplots Generation 2/2")

        for x, hue in product(SEVERITY, TRADITIONAL_LABELS + PROXIES + OUTCOMES):
            pdf.savefig(discrete_multivariate_histogram(dataset, x, hue))
        print("7. Histograms Generation")

        pdf.savefig(lr_abg.plot())
        pdf.savefig(lr_abg.beeswarm(train_set, test_set))
        pdf.savefig(lr_abg.plot_roc())
        print("8. Model Tracing 1/4")

        pdf.savefig(rf_abg.plot())
        pdf.savefig(rf_abg.beeswarm(train_set, test_set))
        pdf.savefig(rf_abg.plot_roc())
        print("9. Model Tracing 2/4")

        pdf.savefig(lr_outcome.plot())
        pdf.savefig(lr_outcome.beeswarm(train_set, test_set))
        pdf.savefig(lr_outcome.plot_roc())
        print("10. Model Tracing 3/4")

        pdf.savefig(rf_outcome.plot())
        pdf.savefig(rf_outcome.beeswarm(train_set, test_set))
        pdf.savefig(rf_outcome.plot_roc())
        print("11. Model Tracing 4/4")


print("12. Report Generation Completed")
