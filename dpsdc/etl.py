import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import tempfile
import json
import os
from tqdm import tqdm
from typing import List, Optional, Callable, Union
import warnings

warnings.filterwarnings("ignore")

CONFIGS = json.load(open(Path(__file__).parent / "config.json", "r"))


def parse_race(values: pd.Series) -> pd.Series:
    """Parses MIMIC-IV self-identified race-ethnicity within the categories identified by Yarnell et al.

    Yarnell, Christopher J., Alistair Johnson, Tariq Dam, Annemijn Jonkman, Kuan Liu, Hannah Wunsch, Laurent Brochard, et al. 2023.
    “Do Thresholds for Invasive Ventilation in Hypoxemic Respiratory Failure Exist? A Cohort Study.”
    American Journal of Respiratory and Critical Care Medicine 207 (3): 271–82.

    Args:
        values (pd.Series): Input Series.

    Returns:
        pd.Series: Parse Series.
    """
    mapped_values = np.empty(values.shape[0]) * np.nan
    mapping = {
        "White": "(WHITE)|(PORTUGUESE)",
        "Hispanic": "(HISPANIC)",
        "Asian": "(ASIAN)",
        "Black": "(BLACK)",
        "Other": "(OTHER)|(SOUTH AMERICAN)|(CARIBBEAN ISLAND)|(NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER)|(AMERICAN INDIAN/ALASKA NATIVE FEDERALLY RECOGNIZED TRIBE)",
    }

    for key, regexp in mapping.items():
        mapped_values = np.where(values.str.contains(regexp), key, mapped_values)
    return pd.Series(mapped_values).replace("nan", "Unknown")


def parse_language(values: pd.Series) -> pd.Series:
    mapped_values = np.empty(values.shape[0]) * np.nan
    mapping = {"English": "(ENGLISH)", "Other": "(\?)"}

    for key, regexp in mapping.items():
        mapped_values = np.where(values.str.contains(regexp), key, mapped_values)
    return mapped_values


def parse_insurance(values: pd.Series) -> pd.Series:
    mapped_values = np.empty(values.shape[0]) * np.nan
    mapping = {
        "Medicaid": "(Medicaid)",
        "Medicare": "(Medicare)",
        "Other": "(Other)",
    }

    for key, regexp in mapping.items():
        mapped_values = np.where(values.str.contains(regexp), key, mapped_values)
    return mapped_values


def parse_careunit(values: pd.Series) -> pd.Series:
    """Parses MIMIC-IV Care Units in groups defined by Yarnell et al.

    Yarnell, Christopher J., Alistair Johnson, Tariq Dam, Annemijn Jonkman, Kuan Liu, Hannah Wunsch, Laurent Brochard, et al. 2023.
    “Do Thresholds for Invasive Ventilation in Hypoxemic Respiratory Failure Exist? A Cohort Study.”
    American Journal of Respiratory and Critical Care Medicine 207 (3): 271–82.

    Args:
        values (pd.Series): Input Series.

    Returns:
        pd.Series: Parsed Series.
    """
    mapped_values = np.empty(values.shape[0]) * np.nan
    mapping = {
        "Medical-Surgical": "(Medical Intensive Care Unit)|(Medical/Surgical Intensive Care Unit)|(Surgical Intensive Care Unit)",
        "Cardiac": "(Cardiac Vascular Intensive Care Unit)|(Coronary Care Unit)",
        "Neuro-Trauma": "(Neuro Intermediate)|(Neuro Stepdown)|(Neuro Surgical Intensive Care Unit)|(Trauma SICU)",
    }

    for key, regexp in mapping.items():
        mapped_values = np.where(values.str.contains(regexp), key, mapped_values)
    return mapped_values


def parse_time(values: pd.Series) -> pd.Series:
    mapped_values = np.empty(values.shape[0]) * np.nan
    mapping = {
        "00-06": (0, 6),
        "06-12": (6, 12),
        "12-18": (12, 18),
        "18-24": (18, 24),
    }

    for key, item in mapping.items():
        values = pd.to_datetime(values)
        mapped_values = np.where(values.dt.hour.between(*item), key, mapped_values)
    return mapped_values.astype(str)


def extract(channel: str, etl: Callable, project_id: str):
    """Pulls data from Big Query.
    It requires for the project to have access to the MIMIC-IV.
    Learn more at

    Args:
        channel (str): _description_
        etl (Callable): _description_
        project_id (str): _description_

    Returns:
        _type_: _description_
    """
    WORKDIR = Path(os.path.join(os.path.dirname(__file__)))
    TEMP = Path(tempfile.gettempdir()) / "dpsdc"
    if not TEMP.is_dir():
        # creates a tmp folder if not existing already
        os.makedirs(TEMP)

    if (TEMP / f"{channel}.csv").is_file():
        # if the cohort was already downlaoded pull it
        __df__ = pd.read_csv(TEMP / f"{channel}.csv")
    else:
        # if firs time running the command, the data is download
        with open(WORKDIR / "procedures" / f"{channel}.sql", "r") as file:
            query = file.read()

        __df__ = pd.read_gbq(query, project_id=project_id)
        __df__.to_csv(TEMP / f"{channel}.csv", index=False)

    df = etl(__df__)
    return df.set_index("stay_id")


def cohort_etl(df: pd.DataFrame) -> pd.DataFrame:
    def received_abg_at_T_etl(df):
        df = df[
            (
                df.time_to_abg__minutes
                <= CONFIGS["received_abg_at_T"]["CAPTURE_WINDOW_UPPER"]
            )
            & (
                df.time_to_abg__minutes
                > CONFIGS["received_abg_at_T"]["CAPTURE_WINDOW_LOWER"]
            )
        ]
        df["received_abg_at_T"] = np.where(
            df.time_to_abg__minutes.astype(float)
            < df.time_to_abg__minutes.astype(float).median(),
            1,
            0,
        )
        df.received_abg_at_T = df.received_abg_at_T.astype("object")
        return df

    def volume_of_abg_etl(df):
        df["abg_rate"] = df["volume_of_abg"] / df["duration__hours"]
        df = df[df.abg_rate.notna()]
        df["abg_rate_pc"] = np.where(df.abg_rate > df.abg_rate.median(), 1, 0)
        df["abg_rate_pc"] = df["abg_rate_pc"].astype(object)
        df = df.drop(["abg_rate"], axis=1)
        return df

    CONFIGS = load_configs()

    # Proxies are defined here.
    if CONFIGS["PROXY"] == ["abg_rate_pc"]:
        df = volume_of_abg_etl(df)
    elif CONFIGS["PROXY"] == ["received_abg_at_T"]:
        df = received_abg_at_T_etl(df)
    else:
        raise ValueError("PROXY not found.")
    df = df.reset_index(drop=True)

    df["race"] = parse_race(df.race)
    df["language"] = parse_language(df.language)
    df["insurance"] = parse_insurance(df.insurance)
    df["careunit"] = parse_careunit(df.first_careunit)
    df["time_of_day"] = parse_time(df.starttime)

    df.los_icu = df.los_icu.astype(float)
    df.admission_age = df.admission_age.astype(float)

    df.hospital_expire_flag = df.hospital_expire_flag.astype("object")

    to_drop = [
        "first_careunit",
        "time_to_abg__minutes",
        "volume_of_abg",
        "admission_location",
        "starttime",
    ]
    to_rename = {
        "hospital_expire_flag": "hospital_death",
        "anchor_year_group": "admission_year",
    }
    df = df.drop(to_drop, axis=1).rename(columns=to_rename)
    return df


def gcs_etl(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(["subject_id"], axis=1).astype("Int64")
    df.stay_id = df.stay_id.astype(int)
    return df


def lab_etl(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(["subject_id"], axis=1).astype(float)
    df.stay_id = df.stay_id.astype(int)
    return df


def vitals_etl(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(["subject_id"], axis=1).astype(float)
    df.columns = [f"capillary_{col}" if "glucose" in col else col for col in df.columns]
    df.stay_id = df.stay_id.astype(int)
    return df


def sofa_etl(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(["subject_id", "hadm_id"], axis=1).astype("Int64")
    df.stay_id = df.stay_id.astype(int)
    return df


def comorbidities_etl(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(["subject_id", "hadm_id"], axis=1)
    temp = df.drop(["stay_id", "charlson_comorbidity_index"], axis=1)
    temp = pd.DataFrame(
        np.where(temp > 0, 1, 0).astype("object"),
        columns=temp.columns,
        index=temp.index,
    )

    df.charlson_comorbidity_index = df.charlson_comorbidity_index.astype("Int64")
    df.stay_id = df.stay_id.astype(int)
    return pd.concat([df[["stay_id", "charlson_comorbidity_index"]], temp], axis=1)


ETLS = {
    "cohort": cohort_etl,
    "gcs": gcs_etl,
    "vitals": vitals_etl,
    "lab": lab_etl,
    "sofa": sofa_etl,
    "comorbidities": comorbidities_etl,
}


def load(channel: str, project_id: str) -> pd.DataFrame:
    return extract(channel, ETLS[channel], project_id)


def begin_workshop(project_id: str):
    TEMP = Path(tempfile.gettempdir()) / "dpsdc"
    PROFILES = {"categorical": [], "ordinal": [], "continuous": []}
    CHANNELS = {}
    pbar = tqdm(ETLS.items())
    for channel, etl in pbar:
        pbar.set_description(channel)
        data = extract(channel, etl, project_id)
        PROFILES["ordinal"] += data.select_dtypes("Int64").columns.to_list()
        PROFILES["continuous"] += data.select_dtypes(float).columns.to_list()
        PROFILES["categorical"] += data.select_dtypes("object").columns.to_list()
        CHANNELS[channel] = data.columns.to_list()

    with open(TEMP / "PROFILES.json", "w") as file:
        json.dump(PROFILES, file)

    with open(TEMP / "CHANNELS.json", "w") as file:
        json.dump(CHANNELS, file)


def load_configs():
    return json.load(open(Path(__file__).parent / "config.json", "r"))


def load_profiles():
    TEMP = Path(tempfile.gettempdir()) / "dpsdc"
    with open(TEMP / "PROFILES.json", "r") as file:
        PROFILES = json.load(file)
    return PROFILES


def load_channels(channels: Optional[List[str]] = None):
    TEMP = Path(tempfile.gettempdir()) / "dpsdc"
    with open(TEMP / "CHANNELS.json", "r") as file:
        CHANNELS = json.load(file)

    if isinstance(channels, list):
        output = []
        [output := output + CHANNELS[channel] for channel in channels]
    else:
        output = CHANNELS

    return output


def load_view(features: Optional[List[str]], project_id: Optional[str]) -> pd.DataFrame:
    channel = load_channels()
    features = set(features)
    sample = []
    for key, item in channel.items():
        features_in_channel = list(features.intersection(item))
        if len(features_in_channel) > 0:
            sample.append(load(key, project_id)[features_in_channel])
    return pd.concat(sample, axis=1, join="inner")


def categorical_features(features):
    return list(set(load_profiles()["categorical"]).intersection(features))


def ordinal_features(features):
    return list(set(load_profiles()["ordinal"]).intersection(features))


def continuous_features(features):
    return list(set(load_profiles()["continuous"]).intersection(features))


def reset_temp():
    TEMP = Path(tempfile.gettempdir()) / "dpsdc"
    shutil.rmtree(TEMP)
