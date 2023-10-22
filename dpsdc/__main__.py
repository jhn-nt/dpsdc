from pathlib import Path
from appdata import AppDataPaths
import json
import pandas as pd
import numpy as np
from itertools import product
from dataclasses import dataclass, replace
from argparse import ArgumentParser
from tqdm import tqdm
import os

from sklearn.model_selection import RepeatedKFold

from numpy.typing import ArrayLike
from typing import List, Union, Tuple, Iterable
from .loaders import load_table_one,load_disparity_axis,load_proxy
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


def compute_ecdf_from_sample(x:ArrayLike,lower:float=.01,upper:float=.99,n_thresholds:int=100)->Tuple[ArrayLike,ArrayLike]:
    """Computes the Empirical Cumulative Distribution Function, ECDF, from a sample x of a continuous random variable.

    Args:
        x (ArrayLike): Sample.
        lower (float, optional): Lower estimation bound expressed as percentile of x. Defaults to .01.
        upper (float, optional): Upper estimation bound expressed as percentile of x. Defaults to .99.
        n_thresholds (int, optional): Number of thresholds where to estimate densities. Defaults to 100.

    Returns:
        Tuple[ArrayLike,ArrayLike]: Tuple with two outputs:
            1. thresholds: Thresholds of x.
            2. ecdf: Densities at thresholds.
    """
    sample_size=x.size
    qmin,qmax=np.quantile(x,[lower,upper])
    thresholds=np.linspace(qmin, qmax,n_thresholds)

    @np.vectorize
    def p_t_greater_than_x(threshold):
        return np.sum(x<threshold)/sample_size

    ecdf=p_t_greater_than_x(thresholds)
    return thresholds,ecdf

def compute_and_interpolate_ecdf_from_sample(*args,n_points:int=100,**kwargs)->Tuple[ArrayLike,ArrayLike]:
    """Computes the Empirical Cumulative Distribution Function, ECDF, from a sample x of a continuous random variable.

    Args:
        x (ArrayLike): Sample.
        lower (float, optional): Lower estimation bound expressed as percentile of x. Defaults to .01.
        upper (float, optional): Upper estimation bound expressed as percentile of x. Defaults to .99.
        n_thresholds (int, optional): Number of thresholds where to estimate densities. Defaults to 100.
        n_points (int, optional): Number of interpolating points.

    Returns:
        Tuple[ArrayLike,ArrayLike]: Tuple with two outputs:
            1. thresholds: Interpolated Thresholds of x.
            2. ecdf: Interpolated Densities at thresholds.
    """
    thresholds,raw_ecdf=compute_ecdf_from_sample(*args,**kwargs)
    eps=1/len(thresholds)
    interpol_ecdf=np.linspace(eps,1-eps, n_points)
    interpol_thresholds=np.interp(interpol_ecdf,raw_ecdf,thresholds)
    return interpol_thresholds,interpol_ecdf

def run_analysis_between_proxy_and_continuous_disparity_axis(axis:str,DATA_PATH:Path,timestamp_variances__min:Iterable=[0,1,2,3,4,5,6,7,9,10]):

    disparity_axis=load_disparity_axis(DATA_PATH)[axis].values
    proxy=load_proxy(DATA_PATH,timestamp_variance__hours=0).proxy.values
    timestamp_variance=np.asarray(timestamp_variances__min)/60
    cv=RepeatedKFold()

    n_cycles=cv.get_n_splits()*len(timestamp_variance)

    @dataclass
    class Accumulate:
        t:List[ArrayLike]
        ecdf: List[ArrayLike]

        @classmethod
        def empty(cls):
            return cls(t=[],ecdf=[])

        @classmethod
        def from_output(cls, t,ecdf):
            return cls(t=[t],ecdf=[ecdf])
    
        def __add__(self,other):
            return replace(self,t=self.t+other.t,ecdf=self.ecdf+other.ecdf)
        
        def compute(self):
            return np.stack(self.t),np.stack(self.ecdf)


    axis_m=Accumulate.empty()
    proxy_m=Accumulate.empty() 
    bias_and_slope=[]

    for tv,(_,test) in tqdm(product(timestamp_variance,cv.split(disparity_axis,proxy)),total=n_cycles):
        adjusted_proxy=proxy + np.random.normal(0,tv)
        axis_m+=Accumulate.from_output(*compute_and_interpolate_ecdf_from_sample(disparity_axis[test]))
        proxy_m+=Accumulate.from_output(*compute_and_interpolate_ecdf_from_sample(adjusted_proxy[test]))

        bias_and_slope.append(np.polyfit(axis_m.t[-1],proxy_m.t[-1],1))

    return {
        axis:axis_m.compute(),
        "proxy":proxy_m.compute(),
        "bias_and_slope":np.stack(bias_and_slope)}



if __name__ == "__main__":
    download()

    # Building Table One
    df = load_table_one(DATA_PATH)
    table_one = TableOne(df, categorical=categorical(df), groupby="proxy", pval=True)

    # Saving Results
    table_one.to_excel(OUTPUT_PATH / "table_one.xlsx")
