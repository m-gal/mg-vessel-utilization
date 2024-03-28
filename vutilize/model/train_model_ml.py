"""
    LOAD preprocessed development dataset:
        "containershipdb_vu_devset.csv"
    and SPLIT it onto TRAIN & TEST sets
    with BAYESIAN Cross-Validation Searching algorithm upon model
    FIND best huperparams for model on TRAIN set
    TRAIN ML model w/ hyperparams found out on DEV
    LOG results with MLflow Tracking

    @author: mikhail.galkin
"""


#%% Load libraries
import os
import sys
import winsound
import warnings

import pandas as pd
import sklearn
import mlflow
import xgboost as xgb

from sklearn import model_selection
from pprint import pprint

#%% Load project's stuff
sys.path.extend([".", "./.", "././.", "../..", "../../.."])
os.environ["NUMEXPR_MAX_THREADS"] = "48"
import vutilize

# Load custom classes and utils
from vutilize.config import data_processed_dir
from vutilize.config import models_dir

from vutilize.utils import load_dev_csv
from vutilize.utils import set_pd_options
from vutilize.utils import set_matlotlib_params
from vutilize.utils import reset_pd_options

from vutilize.plots import plot_residuals_errors

from vutilize.model._xgb import model as get_model_xgb
from vutilize.model._xgb import param_search as get_param_search_xgb
from vutilize.model._rf import model as get_model_rf
from vutilize.model._rf import param_search as get_param_search_rf

from vutilize.data.shipdb_train_model_to_enrich import bayesian_search_cv
from vutilize.data.shipdb_train_model_to_enrich import save_params
from vutilize.data.shipdb_train_model_to_enrich import train_model
from vutilize.data.shipdb_train_model_to_enrich import calculate_metrics


#%%! Toggles to flow through
rnd_state = 42

#%% Print out functions --------------------------------------------------------
def print_versions():
    print(f"Packages' versions:")
    print(f"\tPandas: {pd.__version__}")
    print(f"\tScikit-learn: {sklearn.__version__}")
    print(f"\tXGBoost: {xgb.__version__}")
    print(f"\tMLflow: {mlflow.__version__}")


def print_model(model_name, model, model_params):
    print(f" ")
    print(f"model_name: {model_name}")
    pprint(model)
    print(f"model_params:")
    pprint(model_params)


#%% Data -----------------------------------------------------------------------
def get_data(df, target):
    print(f"\nGet train & test data with the target [{target}]:")
    y_dev = df.pop(target)
    X_dev = df
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X_dev,
        y_dev,
        train_size=0.8,
        random_state=rnd_state,
        shuffle=True,
    )
    print(f"X_train: {X_train.shape[1]} variables & {X_train.shape[0]} records.")
    print(f"X_test: {X_test.shape[1]} variables & {X_test.shape[0]} records.")
    return X_dev, y_dev, X_train, y_train, X_test, y_test


#%% Main =======================================================================
def main(
    target,
    file_to_load="containershipdb_vu_devset.csv",
    n_bayes_params_sets=50,
    n_splits=5,
):
    """Performs FITTING MODELS on DEVELOPMENT set with searching an
    optimal hyper-parameters with Baeysian search.

    Args:
        file_to_load (str): name of csv file w/ processed development data
        target (str): target to predict
        * n_bayes_params_sets (int, optional): Number of parameter settings that are sampled.\
            Defaults to 20.
        * n_splits (int, optional): Num. of splits in Cross-Validation strategy.\
            Defaults to 5.
        * log_metrics_residuals (bool, optional): Toogle to calculate and mlflow log model's
        metricd and rediduals.\
            Defaults to True.
    """

    print(f"---------- START: Vessels' Utilization Model Training ------------")
    set_pd_options()
    set_matlotlib_params()
    print_versions()
    # * Data
    df = load_dev_csv(dir=data_processed_dir, file=file_to_load, target=target)
    X_dev, y_dev, X_train, y_train, X_test, y_test = get_data(df, target)

    # * Bayesian optimization
    # Find opt for hypert-params on TRAIN set
    model_pack = get_model_xgb()

    _, _, bayes_space = get_param_search_xgb()
    bayes_space.pop("booster")  # use for XGB
    bayes_space.pop("colsample_bytree")  # use for XGB
    # bayes_space.pop("criterion")  # use for RF

    model_pack_opted, id = bayesian_search_cv(
        X_train,
        y_train,
        model_pack,
        bayes_space,
        n_bayes_params_sets=n_bayes_params_sets,
        n_splits=n_splits,
        log_model=False,
        log_metrics_residuals=True,
    )
    save_params(model_pack_opted, id, models_dir, target)

    # * Train & Evaluate model on TRAIN & unseen TEST sets
    print(f"\n---------- Train & Evaluate model on unseen TEST set ------------")
    model = model_pack[0]
    model_train = model.fit(X_train, y_train)
    residuals_errors_on_test = plot_residuals_errors(
        model_train,
        X_train,
        y_train,
        X_test,
        y_test,
    )
    metrics_on_test = calculate_metrics(model_train, X_test, y_test, "test")

    # * Train final model on whole DEV set
    print(f"\n------------- Train final model on DEV set-----------------------")
    model_dev, mlflow_ids = train_model(
        X_dev,
        y_dev,
        model_pack_opted,
        log_model=True,
        save_model_aside=True,
        set_name="dev",
    )
    # Log the model performance on unseen data
    with mlflow.start_run(experiment_id=mlflow_ids[0], run_id=mlflow_ids[1]) as run:
        # Log prediction errors and residuals plot
        name = f"{type(model_dev).__name__}-{target}_{mlflow_ids[1]}.png"
        path = f"./plots/{name}"
        mlflow.log_figure(residuals_errors_on_test, path)

        # Log metriccs on unseen test data
        mlflow.log_dict(metrics_on_test, "metrics_on_unseen_test.txt")

    reset_pd_options()
    print(f"\n!!! DONE: Train the model !!!")
    winsound.Beep(frequency=3000, duration=300)


#%% RUN ========================================================================
if __name__ == "__main__":  #! Make sure that Tracking Server has been run.
    warnings.filterwarnings("ignore")
    rnd_state = 42
    main(
        target="vu_estimated",
        n_bayes_params_sets=50,
        n_splits=5,
    )

#%% Auxilary cell
# file_to_load = "containershipdb_vu_devset.csv"
# target = "vu_estimated"
# n_bayes_params_sets = 2
# n_splits = 2
# log_metrics_residuals = True
