"""
    LOAD raw "shipdb_export_04_2021.csv" data
    LOAD some preprocessed data:
        "shipdb_042021_X_YYYYYY.csv"
    MERGE sub-DataFrame from raw data with prepocessed dataset by "imo"
    SAVE merged data making it possible to predict missing values

    @author: mikhail.galkin
"""

# %% Load libraries
import sys
import pandas as pd
from pathlib import Path


# %% Load project stuff
sys.path.extend([".", "./.", "././.", "../..", "../../.."])
data_raw_dir = Path("d:/fishtailS3/ls-aishub-inflated/shipdb")

# Load project configu
from vutilize.config import data_processed_dir

# Load project utils
from vutilize.utils import set_pd_options
from vutilize.utils import reset_pd_options
from vutilize.utils import df_get_glimpse
from vutilize.data.shipdb_process_data import load_raw_data
from vutilize.data.shipdb_train_model_to_enrich import load_data as load_prep_data


# %% Add data
def add_columns(df_prep, df_raw, cols_to_add):
    print(f"\nAdd original columns to preprocessed dataset...")
    cols_to_add.append("imo")
    df = pd.merge(df_prep, df_raw[cols_to_add], how="left", on="imo")
    print(f"\tMerged data has {len(df)} rows X {len(df.columns)} cols...")
    return df


# %% Main =======================================================================
def main(
    file_to_load_raw,
    file_to_load_prep,
    file_to_save,
    cols_to_add,
):
    set_pd_options()
    df_raw = load_raw_data(file_to_load_raw)
    df_prep = load_prep_data(file_to_load_prep)
    df_get_glimpse(df_prep)
    df = add_columns(df_prep, df_raw, cols_to_add)
    df_get_glimpse(df)
    df.to_csv(data_processed_dir / file_to_save, header=True, index=False)
    reset_pd_options()
    print(f"!!! DONE !!!")


# %% Workflow ===================================================================
if __name__ == "__main__":
    main(
        file_to_load_raw="shipdb_export_04_2021.csv",
        file_to_load_prep="shipdb_042021_29_light_displacement_tonnes.csv",
        file_to_save="shipdb_042021_30_light_displacement_tonnes.csv",
        cols_to_add=["max_draught"],
    )

# %%
