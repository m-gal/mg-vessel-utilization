""" Contains the functions used across the project.

    @author: mikhail.galkin
"""

#%% Import needed python libraryies and project config info
import os
import sys
import joblib
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import mlflow
from pathlib import Path
from matplotlib import rcParams
from mlflow.tracking import MlflowClient

# from sklearn.metrics import max_error
# from sklearn.metrics import mean_absolute_error
# from sklearn.metrics import mean_squared_error
from IPython.display import display
from pprint import pprint


sys.path.extend([".", "./.", "././.", "../..", "../../.."])
from vutilize.config import data_processed_dir

# ------------------------------------------------------------------------------
# ----------------------------- P A R A M E T E R S ----------------------------
# ------------------------------------------------------------------------------
#%% Set up: Pandas options
def set_pd_options():
    """Set parameters for PANDAS to InteractiveWindow"""

    display_settings = {
        "max_columns": 40,
        "max_rows": 220,
        "width": 500,
        "max_info_columns": 500,
        "expand_frame_repr": True,  # Wrap to multiple pages
        "float_format": lambda x: "%.5f" % x,
        "precision": 5,
        "show_dimensions": True,
    }
    print("Pandas options established are:")
    for op, value in display_settings.items():
        pd.set_option(f"display.{op}", value)
        option = pd.get_option(f"display.{op}")
        print(f"\tdisplay.{op}: {option}")


#%% Set up: Reset Pandas options
def reset_pd_options():
    """Set parameters for PANDAS to InteractiveWindow"""

    pd.reset_option("all")
    print("Pandas all options re-established.")


#%% Set up: Matplotlib params
def set_matlotlib_params():
    """Set parameters for MATPLOTLIB to InteractiveWindow"""

    rcParams["figure.figsize"] = 10, 5
    rcParams["axes.spines.top"] = False
    rcParams["axes.spines.right"] = False
    rcParams["xtick.labelsize"] = 10
    rcParams["ytick.labelsize"] = 11


# ------------------------------------------------------------------------------
# ---------------------- L O A D I N G    S T U F F ----------------------------
# ------------------------------------------------------------------------------
def load_dev_csv(
    dir=data_processed_dir,
    file="containershipdb_vu_devset.csv",
    target="vu_estimated",
    **kwargs,
):
    print(f"\nLoad data:\n\tDir: {dir}\n\tFile: {file}")
    usecols_static = [
        "ais_dim_a",
        "ais_dim_b",
        "ais_dim_c",
        "ais_dim_d",
        "draught_fact",
    ] + [target]
    df = pd.read_csv(
        dir / file,
        header=0,
        usecols=usecols_static,
        **kwargs,
    )
    print(f"\tLoaded {len(df)} rows X {len(df.columns)} cols...")
    return df


def load_model(model_path):
    print(f"\nLoad early trained and saved model...")
    print(f"{model_path}")
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model


def make_example_df(target):
    """Create example df for checking model prediction"""
    print(f"\nCreate example df for checking model [{target}] prediction...")
    # Pipeline fitted on pandas data frame, due that convert to df
    # Get instances
    rows = [0, 600, 800]
    df = load_dev_csv(target=target, nrows=1000).iloc[rows, :]

    # With 1 data example
    example_df_1 = df.iloc[[0], :-1]
    y_true_1 = df.iloc[[0], -1].to_list()
    # With 2 data examples
    example_df_2 = df.iloc[[1, 2], :-1]
    y_true_2 = df.iloc[[1, 2], -1].to_list()

    examples = [(example_df_1, y_true_1), (example_df_2, y_true_2)]
    return examples


# ------------------------------------------------------------------------------
# ----------------------------- M E T R I C S ----------------------------------
# ------------------------------------------------------------------------------
def mape(y_true, y_pred):
    """Mean absolute percentage error regression loss.

    Examples
    --------
    >>> from sklearn.metrics import mean_absolute_percentage_error
    >>> y_true = np.array([3, -0.5, 2, 7])
    >>> y_pred = np.array([2.5, 0.0, 2, 8])
    0.3273809523809524...

    >>> y_true = np.array([1.0, -1.0, 2, -2])
    >>> y_pred = np.array([0.7, -0.7, 1.4, -1.4])
    >>> mean_absolute_percentage_error(y_true, y_pred)
    0.30000000000000004...
    """
    # Epsilon: is an arbitrary small yet strictly positive numbe
    # to avoid undefined results when y is zero
    epsilon = np.finfo(np.float64).eps
    ape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)
    return np.average(ape)


# ------------------------------------------------------------------------------
# -----------------C O L U M N S   M A N I P U L A T I O N S -------------------
# ------------------------------------------------------------------------------
def cols_get_mixed_dtypes(df):
    print(f"Get info about columns w/ mixed types...")
    mixed_dtypes = {
        c: dtype
        for c in df.columns
        if (dtype := pd.api.types.infer_dtype(df[c])).startswith("mixed")
    }
    pprint(mixed_dtypes)
    mixed_dtypes = list(mixed_dtypes.keys())
    return mixed_dtypes


def raws_drop_duplicats(df):
    """Drop fully duplicated rows"""
    n = df.duplicated().sum()  # Number of duplicated rows
    df.drop_duplicates(keep="first", inplace=True)
    df = df.reset_index(drop=True)
    print(f"Droped {n} duplicated rows")
    return df


def cols_reorder(df):
    cols = df.columns.to_list()
    cols.sort()
    df = df[cols]
    return df


def cols_get_na(df):
    """Get: Info about NA's in df"""
    print(f"\n#NA = {df.isna().sum().sum()}\n%NA = {df.isna().sum().sum()/df.size*100}")
    # View NA's through variables
    df_na = pd.concat(
        [df.isna().sum(), df.isna().sum() / len(df) * 100, df.notna().sum()],
        axis=1,
        keys=["# NA", "% NA", "# ~NA"],
    ).sort_values("% NA")
    return df_na


def df_get_glimpse(df, n_rows=4):
    """Returns a glimpse for related DF

    Args:
        df ([type]): original Pandas DataFrame
        rnd_n_rows (int, optional): # randomly picked up rows to show.
            Defaults to 4.
    """
    #%% Get first info about data set
    print(f"\nCount of duplicated rows : {df.duplicated().sum()}")
    print(f"\n----DF: Information about:")
    print(f"{df.info(verbose=True, show_counts=True, memory_usage=True)}")
    print(f"\n----DF: Descriptive statistics:")
    display(df.describe(include=None).round(3).T)
    print(f"\n----DF: %Missing data:")
    display(cols_get_na(df))
    if n_rows is not None:
        print(f"\n----DF: Random {n_rows} rows:")
        display(df.sample(n=n_rows).T)


def cols_coerce_mixed_to_num(df, cols_to_num):
    if cols_to_num is None:
        pass
    else:
        print(f"Coerce 'mixed' type column to numeric...")
        df[cols_to_num] = df[cols_to_num].apply(pd.to_numeric, errors="coerce")
    return df


def cols_coerce_mixed_to_str(df, cols_to_str):
    if cols_to_str is None:
        pass
    else:
        print(f"Coerce 'mixed' type column to string...")
        df[cols_to_str] = df[cols_to_str].astype(str)
        # Replace 'nan' with NaN
        df[cols_to_str] = df[cols_to_str].replace({"nan": np.nan})
    return df


def cols_coerce_mixed_to_datetime(df, cols_to_datetime):
    if cols_to_datetime is None:
        pass
    else:
        print(f"Coerce 'mixed' type column to datetime...")
        for col in cols_to_datetime:
            print(f"Coerse to_datetime: {col}")
            df[col] = pd.to_datetime(df[col], errors="raise")
    return df


def cols_cat_to_dummies(df, cols_to_dummies):
    print(f"\nConvert categorical to pd.dummies (OHE)...")
    for col in cols_to_dummies:
        print(f"\tConvert column: {col}")
        dummy = pd.get_dummies(df[[col]])
        df = df.drop(columns=[col])
        df = df.merge(dummy, left_index=True, right_index=True)
    return df


# ------------------------------------------------------------------------------
# -------------------------- U T I L I T I E S ---------------------------------
# ------------------------------------------------------------------------------
def get_outliers_zscore(df, cols_to_check=None, sigma=3):
    print(f"\nGet columns w/ outliers w/ {sigma}-sigma...")
    if cols_to_check is None:
        cols_to_check = df.columns.drop("imo").tolist()
    else:
        cols_to_check = cols_to_check
    cols = []
    nums = []
    df_out = pd.DataFrame()
    for col in cols_to_check:
        mean = df[col].mean()
        std = df[col].std()
        z = np.abs(df[col] - mean) > (sigma * std)
        num_outliers = z.sum()

        if num_outliers > 0:
            print(f"\t{col}: {num_outliers} ouliers.")
            display(df.loc[z, col])
            cols.append(col)
            nums.append(num_outliers)
            df_out = pd.concat([df_out, z], axis=1)
    display(df_out.sum())
    return df_out


def get_outliers_quantile(df, cols_to_check=None, treshold=0.999):
    print(f"\nGet columns w/ outliers w/ {treshold} quantile treshold...")
    if cols_to_check is None:
        cols_to_check = df.columns.drop("imo").tolist()
    else:
        cols_to_check = cols_to_check
    cols = []
    nums = []
    df_out = pd.DataFrame()
    for col in cols_to_check:
        cutoff = df[col].quantile(treshold)
        q = df[col] > cutoff
        num_outliers = q.sum()

        if num_outliers > 0:
            print(f"\t{col}: cutoff = {cutoff}: {num_outliers} ouliers.")
            display(df.loc[q, col])
            cols.append(col)
            nums.append(num_outliers)
            df_out = pd.concat([df_out, q], axis=1)
    display(df_out.sum())
    return df_out


def ridoff_outliers(df, df_out):
    print(f"\nRid off outliers via z-score..")
    idx = df_out.sum(axis=1) > 0
    df = df[~idx]
    print(f"Totally deleted {sum(idx)} outliers...")
    print(f"Data w/o outliers has: {len(df)} rows X {len(df.columns)} cols")
    return df


# ------------------------------------------------------------------------------
# ----------------------------- M L   F L O W ----------------------------------
# ------------------------------------------------------------------------------
def mlflow_set_server_local(experiment):
    print(f"\n!!! MAKE SURE THAT TRACKING SERVER HAS BEEN RUN !!!\n")

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    exp_info = MlflowClient().get_experiment_by_name(experiment)

    print(f"Current tracking uri: {mlflow.get_tracking_uri()}")
    print(f"Current registry uri: {mlflow.get_registry_uri()}")

    if exp_info:
        exp_id = exp_info.experiment_id
    else:
        exp_id = MlflowClient().create_experiment(experiment)

    return exp_id


def mlflow_set_exp_id(experiment: str):
    print(f"\nSetup MLflow tracking server...")
    exp_id = mlflow_set_server_local(experiment)
    print(f"Experiment ID: {exp_id}")
    return exp_id


def mlflow_get_run_data(run_id):
    import time

    """Fetch params, metrics, tags, and artifacts
    in the specified run for MLflow Tracking
    """
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(run_id).data
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
    start_time = client.get_run(run_id).info.start_time
    start_time = time.localtime(start_time / 1000.0)
    run_data = (data.params, data.metrics, tags, artifacts, start_time)
    return run_data


def mlflow_del_default_experiment():
    print(f"\nDelete 'Default' experiment from MLflow loggs...")
    try:
        default_id = mlflow.get_experiment_by_name("Default").experiment_id
        default_loc = mlflow.get_experiment_by_name("Default").artifact_location
        mlflow.delete_experiment(default_id)
        print(f"'Default' experiment located:'{default_loc}' was deleted.\n")
    except Exception as e:
        print(f"'Default' experiment doesnt exist: {e}\n")
    return True


def mlflow_set_registry_local(model_name, tracking_dir, registry_db="mlregistry.db"):
    ## Setup the registry server URI.
    # $ By deafult "sqlite:///mlregistry.db" it create <mlregistry.db>
    # $ and [mlruns] folder in parent irectory
    registry_uri = f"sqlite:///{tracking_dir}\{registry_db}"
    mlflow.tracking.set_tracking_uri(registry_uri)

    # Set given experiment as active experiment.
    # If experiment does not exist, create an experiment with provided name.
    exp_id = mlflow.set_experiment(experiment_name=model_name)

    # The URIs should be different
    # assert mlflow.get_tracking_uri() != mlflow.get_registry_uri()

    def print_mlflow_manual():
        #%% Print out
        print(f"TO RUN MLflow Tracking Server locally see README in ./models")
        print(f"Current tracking uri: {mlflow.get_tracking_uri()}")
        print(f"Current registry uri: {mlflow.get_registry_uri()}")

    print_mlflow_manual()
    return exp_id


def mlflow_set_tracking_local(model_name, tracking_dir):
    ## Setup the tracking server URI.
    ## Should be like "file:///...." and MUST BE NAMED AS 'mlruns'
    tracking_uri = f"file:///{tracking_dir}\mlruns"
    mlflow.tracking.set_tracking_uri(tracking_uri)

    # Set given experiment as active experiment.
    # If experiment does not exist, create an experiment with provided name.
    exp_id = mlflow.set_experiment(experiment_name=model_name)
    return exp_id
