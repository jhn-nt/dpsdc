import pandas as pd
from appdata import AppDataPaths
from pathlib import Path
from typing import List, Tuple
from .feature_parsers import *
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

def __cohort__(DATA_PATH):
    from_cohort=pd.read_pickle(DATA_PATH / "cohort.pkl").stay_id.unique()
    from_proxy=pd.read_pickle(DATA_PATH / "proxy.pkl").stay_id.unique()
    return list(set(from_cohort).intersection(from_proxy))


def load_disparity_axis(DATA_PATH:Path) -> pd.DataFrame:
    """Returns the disparity axis, defined as race, insurance, gender, weight, language a nd BMI,
    for all patients requested by the user.

    Args:
        DATA_PATH (Path): Path to the Data.

    Returns:
        pd.DataFrame: Data.
    """
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
    return __raw_df__.set_index("stay_id")[features].loc[__cohort__(DATA_PATH)]


def load_baselines(DATA_PATH:Path) -> pd.DataFrame:
    """Returns baseline ICU admission charachteristics of each patient.
    Examples are admission_age, SODA score or Charlson Comorbidity Index.

    Args:
        DATA_PATH (Path): Path to the Data.

    Returns:
        pd.DataFrame: Data.
    """
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
    return df.loc[__cohort__(DATA_PATH)]


def load_outcomes(DATA_PATH: Path) -> pd.DataFrame:
    """Returns patients' outcomes.


    Args:
        DATA_PATH (Path): Path to the Data.

    Returns:
        pd.DataFrame: Data.
    """
    __raw_df__ = (
        pd.read_pickle(DATA_PATH / "cohort.pkl")
        .set_index("stay_id")["hospital_expire_flag"]
        .astype(str)
    )
    __raw_df__ = __raw_df__.rename("outcome").replace(
        {"0": "Survivor", "1": "Deceased"}
    )
    return __raw_df__.to_frame().loc[__cohort__(DATA_PATH)]


def load_proxy(DATA_PATH:Path) -> pd.DataFrame:
    """Returns the average time interval between instances of the proxy as well as the number of days the proxy was received.

    Args:
        DATA_PATH (Path): Path to the Data.

    Returns:
        pd.DataFrame: Data.
    """
    __raw_df__ = pd.read_pickle(DATA_PATH / "proxy.pkl")
    __raw_df__ = (
        __raw_df__.groupby("stay_id")
        .average_item_interval.agg(["mean", "count"])
        .rename(columns={"mean": "proxy", "count": "days"})
    )
    return __raw_df__.loc[__cohort__(DATA_PATH)]


def load_proxy_quantiles(DATA_PATH:Path,quantiles: List[str]) -> pd.DataFrame:
    """Returns the average time interval between instances of the proxy as well as the number of days the proxy was received.
    The proxy variables is categorized in different quantiles, as decided by the user, prior to be returned.

    Args:
        DATA_PATH (Path): Path to the Data.
        quantiles (List[str]): Quantile catrgories of the proxy.

    Returns:
        pd.DataFrame: Data.
    """
    proxy_and_days = load_proxy(DATA_PATH)
    proxy_and_days.proxy = pd.qcut(proxy_and_days.proxy, q=quantiles).astype(str)
    return proxy_and_days.loc[__cohort__(DATA_PATH)]


def load_proxy_time_series(DATA_PATH:Path) -> pd.DataFrame:
    """Returns raw daily time series of the proxies-

    Args:
        DATA_PATH (Path): Path to the Data.

    Returns:
        pd.DataFrame: Data.
    """
    return pd.read_pickle(DATA_PATH / "proxy.pkl").set_index("stay_id").loc[__cohort__(DATA_PATH)]


def load_table_one(DATA_PATH:Path) -> pd.DataFrame:
    """Returns all the variables to be dispalyed in the table one. 

    Args:
        DATA_PATH (Path): Path to the Data.

    Returns:
        pd.DataFrame: Data.
    """
    disparity_axis_df = load_disparity_axis(DATA_PATH)
    baselines_df = load_baselines(DATA_PATH)
    outcomes_df = load_outcomes(DATA_PATH)
    proxy_df = load_proxy_quantiles(DATA_PATH,[0, 0.25, 0.5, 0.75, 1])

    return pd.concat(
        [disparity_axis_df, baselines_df, proxy_df, outcomes_df], axis=1, join="inner"
    ).loc[__cohort__(DATA_PATH)]
