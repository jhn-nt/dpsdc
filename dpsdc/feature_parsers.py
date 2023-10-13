import pandas as pd
import numpy as np


def parse_admission_location(values: pd.Series):
    temp = values.str.title()
    temp = np.where(
        temp.isin(["Emergency Room", "Physician Referral", "Transfer From Hospital"]),
        temp,
        "Other",
    )
    return temp


def parse_gender(values: pd.Series) -> pd.Series:
    mapped_values = np.empty(values.shape[0]) * np.nan
    mapping = {
        "Male": "M",
        "Female": "F",
    }
    for key, regexp in mapping.items():
        mapped_values = np.where(values.str.contains(regexp), key, mapped_values)
    return pd.Series(mapped_values).replace("nan", "Other")


def parse_race(values: pd.Series) -> pd.Series:
    """Parses MIMIC-IV self-identified race-ethnicity within the categories identified by Yarnell et al.

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
    """Parses MIMIC-IV patients' spoken language.

    Args:
        values (pd.Series): Input Series.

    Returns:
        pd.Series: Parsed Series.
    """
    mapped_values = np.empty(values.shape[0]) * np.nan
    mapping = {"English": "(ENGLISH)", "Other": "(\?)"}

    for key, regexp in mapping.items():
        mapped_values = np.where(values.str.contains(regexp), key, mapped_values)
    return mapped_values


def parse_insurance(values: pd.Series) -> pd.Series:
    """Parses MIMIC-IV patients'Insurance.

    Args:
        values (pd.Series): Input Series.

    Returns:
        pd.Series: Parsed Series.
    """
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
    """Parses MIMIC-IV patients'time of intervention according to Yarnell et al.

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
        "00-06": (0, 6),
        "06-12": (6, 12),
        "12-18": (12, 18),
        "18-24": (18, 24),
    }

    for key, item in mapping.items():
        values = pd.to_datetime(values)
        mapped_values = np.where(values.dt.hour.between(*item), key, mapped_values)
    return mapped_values.astype(str)


def parse_bmi(values: pd.Series) -> str:
    """BMI Classification.
    Source: Weir, Connor B., and Arif Jan. "BMI classification percentile and cut off points." (2019).

    Args:
        bmi_and_race (pd.Series): A series with at leastan index called bmi and one called race.

    Returns:
        str: BMI classification.
    """

    temp = values.copy()
    temp["bmi"] = values.weight * ((values.height.astype(float) / 100) ** -2)

    def __parse__(bmi_and_race):
        bmi = bmi_and_race.bmi
        ethnicity = bmi_and_race.race

        if bmi < 16.5:
            output = "Severly Underweight"
        elif bmi >= 16.5 and bmi < 18.5:
            output = "Underweight"
        elif bmi >= 18.5:
            if "asian" in ethnicity.lower():
                if bmi >= 18.5 and bmi < 23.0:
                    output = "Normal"
                elif bmi >= 23.0 and bmi < 25.0:
                    output = "Overweight"
                elif bmi >= 25.0:
                    output = "Obesity"
            else:
                if bmi >= 18.5 and bmi < 25.0:
                    output = "Normal"
                elif bmi >= 25.0 and bmi < 30.0:
                    output = "Overweight"
                elif bmi >= 30.0:
                    output = "Obesity"
        else:
            output = "Unknown or Unavailable"
        return output

    return temp.apply(__parse__, axis=1)
