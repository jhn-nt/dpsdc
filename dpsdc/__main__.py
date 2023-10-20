from pathlib import Path
from appdata import AppDataPaths
import json
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
import os

from typing import List, Union
from .loaders import load_table_one
from tableone import TableOne
from hashlib import sha256


parser = ArgumentParser()
parser.add_argument("-d", "--dir", action="store", default="./output")
parser.add_argument("-p", "--project-id", action="store", required=True)
parser.add_argument("-c", "--cohort", action="store", default=f"{str(Path(__file__).parent)}/experiments/turnings/criteria.sql")
parser.add_argument("-dp", "--proxy", action="store", default=f"{str(Path(__file__).parent)}/experiments/turnings/proxy.sql")
args = parser.parse_args()

APP_PATH = AppDataPaths("dpsdc")
APP_PATH.setup()


OUTPUT_PATH = Path(args.dir)
COHORT_PATH = Path(args.cohort)
PROXY_PATH = Path(args.proxy)
PROCEDURES_PATH = Path(__file__).parent / "procedures"

TARGET_FOLDER = sha256((str(COHORT_PATH) + str(PROXY_PATH)).encode()).hexdigest()
DATA_PATH = Path(APP_PATH.app_data_path) / TARGET_FOLDER
PROJECT_ID = args.project_id


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
    download()

    # Building Table One
    df = load_table_one(DATA_PATH)
    table_one = TableOne(df, categorical=categorical(df), groupby="proxy", pval=True)

    # Saving Results
    table_one.to_excel(OUTPUT_PATH / "table_one.xlsx")
