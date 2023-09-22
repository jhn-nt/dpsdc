from etl import (
    begin_workshop,
    load_view,
    categorical_features,
    load_channels,
    load_configs,
)
from models import LogisticRegressionModel, RandomForestModel

from tableone import TableOne
import json
import os

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.preprocessing import OneHotEncoder


from dotenv import dotenv_values
from pathlib import Path
import argparse
from itertools import product

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", action="store", default="./output")
parser.add_argument(
    "-p",
    "--project-id",
    required=True,
    help="Google Cloud project id to run the queries from.",
)
args = parser.parse_args()


CONFIGS = load_configs()

STATS_FEATURES = (
    CONFIGS["OUTCOMES"]
    + CONFIGS["PROXY"]
    + CONFIGS["TRADITIONAL_LABELS"]
    + CONFIGS["VARIABLES"]
)
COVARIATES = load_channels(CONFIGS["COVARIATES_CHANNELS"])
ALL_FEATURES = STATS_FEATURES + COVARIATES

OUTPUT_PATH = Path(args.dir)
PROJECT_ID = args.project_id


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


def relative_frequency_matrix(dataset: pd.DataFrame):
    ohe = OneHotEncoder()
    ohe_data = ohe.fit_transform(dataset).todense()

    mc = []
    for ix, _ in RepeatedKFold().split(ohe_data):
        data = ohe_data[ix, :]
        n_i_j = np.matmul(data.T, data)
        norm = np.broadcast_to(np.diag(n_i_j), n_i_j.shape)
        mc.append(np.expand_dims(n_i_j / norm, axis=0))

    mc = np.concatenate(mc, axis=0)
    corr_mean = pd.DataFrame(
        mc.mean(axis=0),
        index=ohe.get_feature_names_out(),
        columns=ohe.get_feature_names_out(),
    )
    corr_std = pd.DataFrame(
        mc.std(axis=0),
        index=ohe.get_feature_names_out(),
        columns=ohe.get_feature_names_out(),
    )

    fig, ax = plt.subplots(figsize=(15, 15))
    _ = ax.imshow(corr_mean)
    _ = ax.set_xticks(np.arange(corr_mean.shape[0]))
    _ = ax.set_yticks(np.arange(corr_mean.shape[0]))
    _ = ax.set_xticklabels(corr_mean.columns, rotation=90)
    _ = ax.set_yticklabels(corr_mean.columns)
    _ = ax.set_title("Relative Frequency Matrix")

    for i, j in product(range(corr_mean.shape[0]), range(corr_mean.shape[0])):
        if corr_mean.iloc[i, j] < 0.25:
            color = "w"
        else:
            color = "k"

        _ = ax.text(
            j,
            i,
            f"{100*corr_mean.iloc[i,j]:.1f}%",
            ha="center",
            va="center",
            color=color,
        )

    return fig


if __name__ == "__main__":
    if not OUTPUT_PATH.is_dir():
        os.makedirs(OUTPUT_PATH)

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

        pdf.savefig(
            relative_frequency_matrix(dataset[categorical_features(STATS_FEATURES)])
        )
        print("12. relative Frequency")


print("13. Report Generation Completed")
