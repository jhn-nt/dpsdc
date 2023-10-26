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

from .loaders import load_table_one, load_proxy, load_disparity_axis
from .quantities import UnivariateAnalysis





parser = ArgumentParser()
parser.add_argument("-d", "--dir", action="store", default="./output")
parser.add_argument("-p", "--project-id", action="store", required=True)
parser.add_argument("-c", "--cohort", action="store", default=f"{str(Path(__file__).parent)}/experiments/turnings/criteria.sql")
parser.add_argument("-dp", "--proxy", action="store", default=f"{str(Path(__file__).parent)}/experiments/turnings/proxy.sql")
args = parser.parse_args()

APP_PATH = AppDataPaths("dpsdc") # Initializing temp folders
APP_PATH.setup()


OUTPUT_PATH = Path(args.dir) # Output directory for figures and tables
COHORT_PATH = Path(args.cohort) # Path to .sql defining the cohort
PROXY_PATH = Path(args.proxy) # Path to .sql defining the proxy 
PROCEDURES_PATH = Path(__file__).parent / "procedures" # Path to dpsdc internal procedures

TARGET_FOLDER = sha256((str(COHORT_PATH) + str(PROXY_PATH)).encode()).hexdigest() # Unique Name defined from the cohort and proxy
DATA_PATH = Path(APP_PATH.app_data_path) / TARGET_FOLDER # Data and temporary files will be stored here
PROJECT_ID = args.project_id # User defined google project id used to access BigQuery
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


def categorical(df):
    return df.columns[df.dtypes == "object"].to_list()


def continuous(df):
    return df.columns[df.dtypes != "object"].to_list()



if __name__ == "__main__":
    # Downloading the data from BigQuery, if not already available
    download()

    # Building Table One
    df = load_table_one(DATA_PATH)
    table_one = TableOne(df, categorical=categorical(df), groupby="proxy", pval=True)

    # Computing Quantile Maps
    proxy_df=load_proxy(DATA_PATH)
    disparities_df=load_disparity_axis(DATA_PATH)
    experiment=UnivariateAnalysis(
        proxy_name="Turning",
        disparities_axis_name="Weight",
        disparities_axis_uom="Kg(s)",
        protocol__hours=2)
    results=experiment.estimate_quantile_mappings_between_proxy_and_disparity_axis(proxy_df.proxy,disparities_df["weight"])
    quantile_plot=experiment.plot(*results)

    # Saving Results
    table_one.to_excel(OUTPUT_PATH / "table_one.xlsx")
    quantile_plot.savefig(OUTPUT_PATH / "quantile_plot.png",dpi=500)
