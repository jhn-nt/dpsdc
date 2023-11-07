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

from .loaders import load_table_one, load_proxy, load_disparity_axis, load_baselines
from .quantities import UnivariateAnalysis, MultivariateAnalysis
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
    # 1. Downloading the data from BigQuery, if not already available
    download()

    # 2. Building Table One
    df = load_table_one(DATA_PATH)
    table_one = TableOne(df, categorical=categorical(df), groupby="proxy", pval=True)

    # 3. Computing Quantile Maps with a high resolution (100pts) for plots and stats
    proxy_df = load_proxy(DATA_PATH)
    disparities_df = load_disparity_axis(DATA_PATH)
    experiment = UnivariateAnalysis(
        proxy_name="Turning",
        disparities_axis_name="Weight",
        disparities_axis_uom="Kg(s)",
        protocol__hours=2,
        n_points=100,
    )
    results = experiment.estimate_quantile_mappings_between_proxy_and_disparity_axis(
        proxy_df.proxy, disparities_df["weight"]
    )
    slopes = experiment.test_null_hypothesis_that_observed_quantile_mapping_adheres_to_protocol(
        *results
    )
    quantile_plot, slopes_plot, proxy_ecdf_plot, disparity_ecdf_plot = experiment.plot(
        results, slopes
    )

    # 4. Computing Quantile Maps at low resolution (10pts) for descriptive tables
    experiment = UnivariateAnalysis(
        proxy_name="Turning",
        disparities_axis_name="Weight",
        disparities_axis_uom="Kg(s)",
        protocol__hours=2,
        n_points=10,
    )
    results = experiment.estimate_quantile_mappings_between_proxy_and_disparity_axis(
        proxy_df.proxy, disparities_df["weight"]
    )
    univariate_results_df = experiment.to_df(*results)

    # 5. Quantile regression to adjust for confounders
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
    observed_predicted_quantiles_plot = experiment.plot_observed_predicted_quantiles(
        results
    )
    fi_plots_per_model = experiment.plot_fi_boxplots(results)
    shap_plots_per_model = experiment.plot_shapvalues(results)
    test_scores, train_scores, fi_per_model = experiment.to_df(results)

    # Saving Results
    table_one.to_excel(OUTPUT_PATH / "table_one.xlsx")
    univariate_results_df.to_excel(OUTPUT_PATH / "univariate_results.xlsx")
    quantile_plot.savefig(OUTPUT_PATH / "quantile_plot.png", dpi=500)
    slopes_plot.savefig(OUTPUT_PATH / "slopes_plot.png", dpi=500)
    proxy_ecdf_plot.savefig(OUTPUT_PATH / "proxy_ecdf_plot.png", dpi=500)
    disparity_ecdf_plot.savefig(OUTPUT_PATH / "disparity_ecdf_plot.png", dpi=500)
    test_scores.to_excel(OUTPUT_PATH / "test_scores.xlsx")
    train_scores.to_excel(OUTPUT_PATH / "train_scores.xlsx")
    observed_predicted_quantiles_plot.savefig(
        OUTPUT_PATH / "observed_predicted_quantiles_plot.png", dpi=500
    )

    for model_name, fi in fi_per_model.items():
        fi.to_excel(OUTPUT_PATH / f"{model_name}.xlsx")

    for model_name, fig in fi_plots_per_model.items():
        fig.savefig(OUTPUT_PATH / f"{model_name}_fi_plot.png", dpi=500)

    for model_name, fig in shap_plots_per_model.items():
        fig.savefig(OUTPUT_PATH / f"{model_name}_shap_plot.png", dpi=500)
