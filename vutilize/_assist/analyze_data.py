"""
    Make manual analysis
    @author: mikhail.galkin
"""
#%% Load libraries
import os
import sys
import numpy as np
import pandas as pd
import pandas_profiling as pp
import sweetviz as sv
import random

from pathlib import Path
from IPython.display import display
from pprint import pprint


#%% Load project stuff
sys.path.extend([".", "./.", "././.", "../..", "../../.."])

# Load project configu
from vutilize.config import data_raw_dir

# Load project utils
from vutilize.utils import set_pd_options
from vutilize.utils import reset_pd_options

set_pd_options()

#%% LOCAL: First testing loading -----------------------------------------------
data_dir = Path("d:/fishtailS3/ls-aishub-inflated/vessel_tracker_data/")

imo = 9778820  # MILAN MAERSK
file_csv = "vt_1819_9778820.csv"

# file_csv = "vt2018_with_distances.csv"
# file_csv = "vt2019_with_distances.csv"

# file_zip = "export_1610463874875_2106_2018-01-01T000000-2020-01-01T000000.zip"
file_zip = "export_1618833108558_2187_2018-01-01T000000-2019-12-31T000000.zip"

#%% Load
def load_csv(file_csv=file_csv):
    """Pre-loading testing of load"""
    ## num of records in file (excludes header)
    n_len = sum(1 for line in open(data_raw_dir / file_csv)) - 1
    print(f"Original file {file_csv} contain {n_len} rows...")
    df = pd.read_csv(
        data_dir / file_csv,
        nrows=None,
        index_col=0,
    )
    print(f"{file_csv}: \n\tLoaded {len(df)} rows X {len(df.columns)} cols...")
    display(df.sample(n=10, random_state=42).T)
    return df


def load_zip(file_zip=file_zip):
    """Pre-loading testing of load"""
    df = pd.read_csv(
        data_dir / file_zip,
        nrows=1000,
        compression="zip",
        # index_col=0,
    )
    print(f"{file_zip}: \n\tLoaded {len(df)} rows X {len(df.columns)} cols...")
    display(df.sample(n=10, random_state=42).T)
    return df


#%%
def cols_date_in_name(df):
    ## Possible datetime columns
    cols_dates = [col for col in list(df.columns) if "date" in col]
    cols_dates.extend([col for col in list(df.columns) if "time" in col])
    print(f"Columns w/ 'date' in name...")
    display(cols_dates)
    display(df[cols_dates].T)


#%%
def get_imo(df, imo=9778820):
    df = df[df["imo"] == imo]
    display(df)
    return df


#%%
def pickup_immo(df_18, df_19, imo):
    df8 = df_18[df_18["imo"] == imo]
    df8["year"] = 2018
    df9 = df_19[df_19["imo"] == imo]
    df9["year"] = 2019
    df = pd.concat([df8, df9])
    return df


#%%
def save(df):
    df.to_csv(data_raw_dir / "vt_1819_9778820.csv")
