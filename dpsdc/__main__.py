from pathlib import Path
from appdata import AppDataPaths
import json
import os
from tqdm import tqdm


import pandas as pd
import numpy as np

from itertools import product
from dataclasses import dataclass, replace
from argparse import ArgumentParser

from tableone import TableOne
from hashlib import sha256

from .loaders import (
    load_table_one,
    load_proxy,
    load_disparity_axis,
    load_baselines,
    load_proxy_time_series,
)
from .quantities import UnivariateAnalysis, MultivariateAnalysis, ExploatoryAnalysis
from .utils import categorical, continuous


parser = ArgumentParser()
parser.add_argument("-d", "--dir", action="store", default="./output")
parser.add_argument("-p", "--project-id", action="store", required=True)
parser.add_argument(
    "-c",
    "--cohort",
    action="store",
    default=f"{str(Path(__file__).parent)}/experiments/turnings/criteria.sql",
)
parser.add_argument(
    "-dp",
    "--proxy",
    action="store",
    default=f"{str(Path(__file__).parent)}/experiments/turnings/proxy.sql",
)
parser.add_argument("--dry", action="store_true")
args = parser.parse_args()

APP_PATH = AppDataPaths("dpsdc")  # Initializing temp folders
APP_PATH.setup()


OUTPUT_PATH = Path(args.dir)  # Output directory for figures and tables
COHORT_PATH = Path(args.cohort)  # Path to the .sql file defining the cohort
PROXY_PATH = Path(args.proxy)  # Path to the .sql file defining the proxy
PROCEDURES_PATH = (
    Path(__file__).parent / "procedures"
)  # Path to dpsdc template procedures

TARGET_FOLDER = sha256(
    (str(COHORT_PATH) + str(PROXY_PATH)).encode()
).hexdigest()  # Unique Temp folder name defined from the cohort and proxy file names
DATA_PATH = (
    Path(APP_PATH.app_data_path) / TARGET_FOLDER
)  # Data and temporary files will be stored here
PROJECT_ID = args.project_id  # User defined google project id used to access BigQuery
DRY_RUN = args.dry


if DRY_RUN:
    print("DRY RUN")
    print(DATA_PATH)


if not OUTPUT_PATH.is_dir():
    os.mkdir(OUTPUT_PATH)


def download():
    if not DATA_PATH.is_dir():
        os.mkdir(DATA_PATH)
        cohort_criteria = open(COHORT_PATH, "r").read()
        proxy_query = open(PROXY_PATH, "r").read()

        template_query = open(PROCEDURES_PATH / "cohort.sql", "r").read()
        query = template_query.format(cohort_criteria)
        pd.read_gbq(query, project_id=PROJECT_ID).to_pickle(DATA_PATH / "cohort.pkl")
        pd.read_gbq(proxy_query, project_id=PROJECT_ID).to_pickle(
            DATA_PATH / "proxy.pkl"
        )

        pbar = tqdm((PROCEDURES_PATH / "baselines").glob("*.sql"), desc="baselines")
        for procedure in pbar:
            pbar.set_postfix({"procedure": procedure.name})
            template_query = open(procedure, "r").read()
            query = template_query.format(cohort_criteria)
            pd.read_gbq(query, project_id=PROJECT_ID).to_pickle(
                DATA_PATH / f"{procedure.name.split('.sql')[0]}.pkl"
            )


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    # 1. Downloading the data from BigQuery, if not already available
    download()
    print("1. Data Downloaded")

    # 2. Building Table One
    df = load_table_one(DATA_PATH)
    table_one = TableOne(df, categorical=categorical(df), groupby="proxy", pval=True)
    print("2. Table One Created")

    # 3. Computing Quantile Maps with a high resolution (100pts) for plots and stats
    disparity_axis_df = load_disparity_axis(DATA_PATH)
    proxies_df = load_proxy(DATA_PATH)
    experiment = UnivariateAnalysis(
        proxy_name="Turning",
        disparities_axis_name="Weight",
        disparities_axis_uom="Kg(s)",
        protocol__hours=2,
    )
    

    scores = experiment.run(disparity_axis_df.weight.values, proxies_df.proxy.values)
    regression_plots = experiment.plot_regression_by_variance(scores)
    ecdf_plots = experiment.plot_ecdf_by_variance(scores)
    regression_table, fisher_tests = experiment.to_df(scores)
    print("3. Univariate Analysis Concluded")

    # 4. Quantile regression to adjust for confounders
    baseline_df = load_baselines(DATA_PATH)
    disparity_axis_df = load_disparity_axis(DATA_PATH)
    proxies_df = load_proxy(DATA_PATH)
    X_y = pd.concat([baseline_df, disparity_axis_df, proxies_df], axis=1)

    experiment = MultivariateAnalysis(
        proxy_name="Turning",
        disparities_axis_name="Weight",
        disparities_axis_uom="Kg(s)",
        dry=DRY_RUN,
    )
    results = experiment.run(X_y)
    qq_plots_per_model = experiment.plot_observed_predicted_quantiles(results)
    fi_plots_per_model = experiment.plot_fi_boxplots(results)
    shap_plots_per_model = experiment.plot_shapvalues(results)
    test_scores, train_scores, fi_per_model = experiment.to_df(results)
    print("4. Multivariate Analysis Concluded")

    # 5. Descriptive plots
    time_series_df = load_proxy_time_series(DATA_PATH)
    experiment = ExploatoryAnalysis(
        proxy_name="Turning",
        disparities_axis_name="Weight",
        disparities_axis_uom="Kg(s)",
        protocol__hours=2,
    )
    boxplots_per_disparity = experiment.boxplot_by_features(
        proxies_df.proxy, disparity_axis_df
    )
    trends_per_disparity = experiment.plot_daily_trends(
        time_series_df, disparity_axis_df
    )
    variance_effect_fig = experiment.plot_timestamp_variance_effect(proxies_df.proxy)
    print("5. Plotting")

    # 6. Saving Results
    table_one.to_excel(OUTPUT_PATH / "table_one.xlsx")
    regression_table.to_excel(OUTPUT_PATH / "univariate_regression_results.xlsx")
    fisher_tests.to_excel(OUTPUT_PATH / "univariate_significance_tests.xlsx")
    test_scores.to_excel(OUTPUT_PATH / "test_scores.xlsx")
    train_scores.to_excel(OUTPUT_PATH / "train_scores.xlsx")
    variance_effect_fig.savefig(OUTPUT_PATH / "variance_effect.png", dpi=500)

    if not (OUTPUT_PATH / "qq_plots").is_dir():
        os.mkdir(OUTPUT_PATH / "qq_plots")
    for model_name, fig in qq_plots_per_model.items():
        fig.savefig(OUTPUT_PATH / "qq_plots" / f"{model_name}.png", dpi=500)


    if not (OUTPUT_PATH / "fi_tables").is_dir():
        os.mkdir(OUTPUT_PATH / "fi_tables")
    for model_name, fi in fi_per_model.items():
        fi.to_excel(OUTPUT_PATH / "fi_tables" / f"{model_name}.xlsx")

    if not (OUTPUT_PATH / "fi_plots").is_dir():
        os.mkdir(OUTPUT_PATH / "fi_plots")
    for model_name, fig in fi_plots_per_model.items():
        fig.savefig(OUTPUT_PATH / "fi_plots" / f"{model_name}_fi_plot.png", dpi=500)

    if not (OUTPUT_PATH / "shap").is_dir():
        os.mkdir(OUTPUT_PATH / "shap")
    for model_name, fig in shap_plots_per_model.items():
        fig.savefig(OUTPUT_PATH / "shap" / f"{model_name}_shap_plot.png", dpi=500)

    if not (OUTPUT_PATH / "boxplots").is_dir():
        os.mkdir(OUTPUT_PATH / "boxplots")
    for disparity_name, fig in boxplots_per_disparity.items():
        fig.savefig(OUTPUT_PATH / "boxplots" / f"{disparity_name}_boxplot.png", dpi=500)

    if not (OUTPUT_PATH / "trends").is_dir():
        os.mkdir(OUTPUT_PATH / "trends")
    for disparity_name, fig in trends_per_disparity.items():
        fig.savefig(OUTPUT_PATH / "trends" / f"{disparity_name}_trend.png", dpi=500)

    if not (OUTPUT_PATH / "regression_plots").is_dir():
        os.mkdir(OUTPUT_PATH / "regression_plots")
    for variance, fig in regression_plots.items():
        fig.savefig(OUTPUT_PATH / "regression_plots" / f"{variance}.png", dpi=500)

    if not (OUTPUT_PATH / "ecdf_plots").is_dir():
        os.mkdir(OUTPUT_PATH / "ecdf_plots")
    for variance, fig in ecdf_plots.items():
        fig.savefig(OUTPUT_PATH / "ecdf_plots" / f"{variance}.png", dpi=500)
    print("6. Done")


