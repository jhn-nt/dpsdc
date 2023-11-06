import pandas as pd
from typing import List


def categorical(df: pd.DataFrame) -> List[str]:
    """Returns the list of features of type 'object' from dataframe df.

    Args:
        df (pd.DataFrame): Input.

    Returns:
        List[str]: List of features.
    """
    return df.columns[df.dtypes == "object"].to_list()


def continuous(df: pd.DataFrame) -> List[str]:
    """Returns the list of features of type 'int' or 'float' from dataframe df.

    Args:
        df (pd.DataFrame): Input.

    Returns:
        List[str]: List of features.
    """
    return df.columns[df.dtypes != "object"].to_list()
