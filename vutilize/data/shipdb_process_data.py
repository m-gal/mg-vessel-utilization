"""
    LOAD raw "shipdb_export_04_2021.csv" data
    MAKE some data transformations
    SAVE data for modeling and enriching in [./project/data/processed].

    @author: mikhail.galkin
"""

#%% Load libraries
import os
import sys
import numpy as np
import pandas as pd

# from scipy import stats
from dataprep.eda import create_report
from pathlib import Path
from IPython.display import display


#%% Load project stuff
sys.path.extend([".", "./.", "././.", "../..", "../../.."])
# os.environ["NUMEXPR_MAX_THREADS"] = "48"

# Load project configu
from vutilize.config import data_processed_dir
from vutilize.config import reports_dir
from vutilize.config import shipdb_cols_to_use

# Load project utils
from vutilize.utils import set_pd_options
from vutilize.utils import reset_pd_options
from vutilize.utils import cols_coerce_mixed_to_num
from vutilize.utils import df_get_glimpse
from vutilize.utils import get_outliers_zscore
from vutilize.utils import ridoff_outliers


#%% Load
def load_raw_data(file_to_load):
    """Load raw dataset from file
    Args:
        dir (pathlib.WindowsPath): Main project's directory
        file_to_load (str, optional): File name to load.
    Returns:
        Pandas DF: [description]
    """
    print(f"\nLoad data from {file_to_load}...")
    dir = Path("d:/fishtailS3/ls-aishub-inflated/shipdb")
    df = pd.read_csv(
        dir / file_to_load,
        header=0,
        usecols=shipdb_cols_to_use,
        low_memory=False,
    )
    print(f"\n{file_to_load}: Loaded {len(df)} rows X {len(df.columns)} cols")

    print(f"Get only container ships data...")
    df = df.loc[df["my_vessel_type.0"] == "container_ship"]
    df = df.drop(["my_vessel_type.0"], axis=1)

    print(f"Found {df.duplicated().sum()} duplicated rows...")
    n = df.duplicated().sum()  # Number of duplicated rows
    df.drop_duplicates(keep="first", inplace=True)
    df = df.reset_index(drop=True)
    print(f"Droped {n} duplicated rows...")

    print(f"Reorder columns...")
    cols = df.columns.to_list()
    cols.remove("imo")
    cols.sort()
    cols.insert(0, "imo")
    df = df[cols]

    print(f"Data w/ container ships has: {len(df)} rows X {len(df.columns)} cols")
    print(f"View random 4 rows:")
    display(df.sample(4))
    return df


#%% Strings
def cols_coerce_str_to_float(df, col="mcr_fuel_consumption"):
    print(f"\nCoerce string to float...")
    df[col] = df[col].str.extract(r"(\d+.\d+)").astype("float")
    return df


#%% Zero values
def get_info_zeros(df):
    print(f"\nGet columns with zeros...")
    zeros = pd.DataFrame((df == 0).sum(), columns=["#Zeros"])
    cols_w_zeros = zeros[zeros["#Zeros"] > 0].index.tolist()
    nans = pd.DataFrame(df.isna().sum(), columns=["#NaNs"])
    zn = zeros.join(nans)
    display(zn[zeros["#Zeros"] > 0])
    return cols_w_zeros


def zeroes_to_nan(df, cols_w_zeros):
    print(f"\nCoerce zeros to NaN...")
    df[cols_w_zeros] = df[cols_w_zeros].replace({0: np.nan})
    return df


#%% Negative values
def get_info_negative(df):
    print(f"\nGet columns with negative values...")
    neg = pd.DataFrame((df < 0).sum(), columns=["#Neg"])
    cols_w_neg = neg[neg["#Neg"] > 0].index.tolist()
    nans = pd.DataFrame(df.isna().sum(), columns=["#NaNs"])
    zn = neg.join(nans)
    display(zn[neg["#Neg"] > 0])
    return cols_w_neg


def negative_to_positive(df, cols_w_neg):
    print(f"\nCoerce negative values to positive...")
    df[cols_w_neg] = df[cols_w_neg].abs()
    return df


#%% Outliers
def get_outliers_quantile(df, treshold=0.9999):
    print(f"\nGet columns w/ outliers w/ {treshold} quantile treshold...")
    cols = []
    nums = []
    for col in df.columns.tolist()[1:]:
        num_of_outliers = len(df[df[col] > df[col].quantile(treshold)])
        if num_of_outliers > 0:
            cutoff = df[col].quantile(treshold)
            print(f"For <{col}> the cutoff is equal {cutoff}.")
            display(df.nlargest(5, [col])[col])
            cols.append(col)
            nums.append(num_of_outliers)
    display(list(zip(cols, nums)))
    cols_w_outl = [col for (col, num) in list(zip(cols, nums)) if num == 1]
    return (cols_w_outl, treshold)


def outliers_quantile_to_nan(df, cols_w_outl):
    print(f"\nCoerce super outliers to NaN...")
    cols = cols_w_outl[0]
    treshold = cols_w_outl[1]
    for col in cols:
        cutoff = df[col].quantile(treshold)
        print(f"For <{col}> the cutoff is equal {cutoff}.")
        display(df.nlargest(5, [col])[col])
        indx = df.nlargest(5, [col])[col].index.tolist()
        df[col].where(df[col] < cutoff, np.NaN, inplace=True)
        display(df.iloc[indx][col])
    return df


#%% Reports
def make_dataprep_report(
    df,
    title,
    name,
):
    report_name = name
    report = create_report(df, title)
    report.save(filename=report_name, to=reports_dir)
    report.show_browser()


#%% Main =======================================================================
def main(
    file_to_load="shipdb_export_04_2021.csv",
    file_to_save="shipdb_042021_0_containers.csv",
):
    set_pd_options()
    df = load_raw_data(file_to_load)
    # Make minimal processing
    df = cols_coerce_mixed_to_num(df, ["cargo_holds", "engine_power"])
    cols_w_zeros = get_info_zeros(df)
    df = zeroes_to_nan(df, cols_w_zeros)
    cols_w_neg = get_info_negative(df)
    df = negative_to_positive(df, cols_w_neg)
    df_out = get_outliers_zscore(
        df,
        cols_to_check=[
            "ballast_water",
            "breadth_extreme",
            "cargo_holds",
            "depth",
            "displacement",
            "freshwater",
            "fuel",
            "fuel_oil",
        ],
        sigma=3,
    )
    df = ridoff_outliers(df, df_out)
    df_get_glimpse(df)
    make_dataprep_report(
        df,
        title="ContainerShipDB 042021 before enriching",
        name="containershipdb_042021_notenriched-dataprep",
    )
    df.to_csv(data_processed_dir / file_to_save, header=True, index=False)
    reset_pd_options()
    print(f"!!! DONE !!!")


#%% Workflow ===================================================================
if __name__ == "__main__":
    main(
        file_to_load="shipdb_export_04_2021.csv",
        file_to_save="shipdb_042021_0_containers.csv",
    )

#%%
