import pandas as pd
from appdata import AppDataPaths
from pathlib import Path
from typing import List, Tuple
from .feature_parsers import *
import warnings

warnings.filterwarnings("ignore")

APP_PATH = AppDataPaths("dpsdc")
DATA_PATH = Path(APP_PATH.app_data_path)


def load_disparity_axis() -> pd.DataFrame:
    __raw_df__ = pd.read_pickle(DATA_PATH / "cohort.pkl")
    __raw_df__["race"] = parse_race(__raw_df__["race"])
    __raw_df__["careunit"] = parse_careunit(__raw_df__["first_careunit"])
    __raw_df__["insurance"] = parse_insurance(__raw_df__["insurance"])
    __raw_df__["language"] = parse_language(__raw_df__["language"])
    __raw_df__["bmi"] = parse_bmi(__raw_df__[["height", "weight", "race"]])
    __raw_df__["gender"] = parse_gender(__raw_df__["gender"])
    __raw_df__["admission_location"] = parse_admission_location(
        __raw_df__["admission_location"]
    )

    __raw_df__["los_icu"] = __raw_df__["los_icu"].astype(float)
    __raw_df__["height"] = __raw_df__["height"].astype(float)
    __raw_df__["weight"] = __raw_df__["weight"].astype(float)
    __raw_df__["admission_age"] = __raw_df__["admission_age"].astype(float)

    features = ["race", "gender", "bmi", "weight", "language", "insurance"]
    return __raw_df__.set_index("stay_id")[features]


def load_baselines() -> pd.DataFrame:
    def demographic():
        __raw_df__ = pd.read_pickle(DATA_PATH / "cohort.pkl")
        __raw_df__["careunit"] = parse_careunit(__raw_df__["first_careunit"])
        __raw_df__["admission_location"] = parse_admission_location(
            __raw_df__["admission_location"]
        )
        __raw_df__["admission_age"] = __raw_df__["admission_age"].astype(float)
        __raw_df__ = __raw_df__.rename(columns={"anchor_year_group": "admission_year"})
        return __raw_df__.set_index("stay_id")[
            ["careunit", "admission_age", "admission_location", "admission_year"]
        ]

    def severity_scores():
        sofa = pd.read_pickle(DATA_PATH / "sofa.pkl").set_index("stay_id")["SOFA"]
        sofa = sofa.rename("sofa").astype(float)

        cci = (
            pd.read_pickle(DATA_PATH / "comorbidities.pkl")
            .set_index("stay_id")["charlson_comorbidity_index"]
            .astype(int)
        )
        cci = cci.rename("cci").astype(float)

        gcs = pd.read_pickle(DATA_PATH / "gcs.pkl").set_index("stay_id")["gcs_min"]
        gcs = gcs.rename("gcs").astype(float)
        return pd.concat([sofa, cci, gcs], axis=1, join="outer")

    df = pd.concat([demographic(), severity_scores()], axis=1, join="outer")
    return df


def load_outcomes() -> pd.DataFrame:
    __raw_df__ = (
        pd.read_pickle(DATA_PATH / "cohort.pkl")
        .set_index("stay_id")["hospital_expire_flag"]
        .astype(str)
    )
    __raw_df__ = __raw_df__.rename("outcome").replace(
        {"0": "Survivor", "1": "Deceased"}
    )
    return __raw_df__.to_frame()


def load_proxy() -> pd.DataFrame:
    __raw_df__ = pd.read_pickle(DATA_PATH / "proxy.pkl")
    __raw_df__ = (
        __raw_df__.groupby("stay_id")
        .average_item_interval.agg(["mean", "count"])
        .rename(columns={"mean": "proxy", "count": "days"})
    )
    return __raw_df__


def load_proxy_quantiles(quantiles: List[str]) -> pd.DataFrame:
    proxy_and_days = load_proxy()
    proxy_and_days.proxy = pd.qcut(proxy_and_days.proxy, q=quantiles).astype(str)
    return proxy_and_days


def load_proxy_visualization() -> pd.DataFrame:
    return load_proxy()


def load_table_one() -> pd.DataFrame:
    disparity_axis_df = load_disparity_axis()
    baselines_df = load_baselines()
    outcomes_df = load_outcomes()
    proxy_df = load_proxy_quantiles([0, 0.25, 0.5, 0.75, 1])

    return pd.concat(
        [disparity_axis_df, baselines_df, proxy_df, outcomes_df], axis=1, join="inner"
    )
