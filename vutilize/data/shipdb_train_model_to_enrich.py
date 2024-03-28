"""
    LOAD minimal preprocessed data:
        "shipdb_042021_0_containers.csv"
    and SPLIT it onto DEV set where target variable is not NaN
    on DEV define the Feature Importance for target w/ default XGBoost
    SELECT from DEV only important features
    with BAYESIAN Cross-Validation Searching algorithm upon model
    FIND best huperparams for model
    TRAIN model w/ huperparams found on DEV
    PREDICT missing value for target variable
    SAVE enriched data
    LOG results with MLflow Tracking

    Repeat through iterating all variables needed.

    Created on April 2021
    @author: mikhail.galkin
"""

# ------------------------------------------------------------------------------
# ----------------- E N R I C H   S H I P D B   M O D E L S --------------------
# ------------------------------------------------------------------------------
#%% Load libraries
import os
import sys
import json
import time
import winsound
import warnings

import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt

import mlflow
import xgboost as xgb
import sweetviz as sv

from dataprep.eda import create_report
from sklearn import model_selection
from sklearn import metrics
from skopt import BayesSearchCV
from pprint import pprint
from IPython.display import display

#%% Load project's stuff
sys.path.extend([".", "./.", "././.", "../..", "../../.."])
import vutilize

# Load custom classes and utils
from vutilize.config import data_processed_dir
from vutilize.config import models_dir
from vutilize.config import reports_dir
from vutilize.utils import set_pd_options
from vutilize.utils import set_matlotlib_params
from vutilize.utils import reset_pd_options
from vutilize.utils import df_get_glimpse
from vutilize.utils import mape as my_mape
from vutilize.utils import mlflow_set_exp_id
from vutilize.utils import mlflow_get_run_data
from vutilize.utils import mlflow_del_default_experiment
from vutilize.plots import plot_residuals_errors

from vutilize.model._xgb import model as get_model_xgb
from vutilize.model._xgb import param_search as get_param_search_xgb
from vutilize.model._rf import model as get_model_rf
from vutilize.model._rf import param_search as get_param_search_rf

#%%! Toggles to flow through
rnd_state = 42

#%% Print out functions --------------------------------------------------------
def print_versions():
    print(f"Packages' versions:")
    print(f"\tNumpy: {np.__version__}")
    print(f"\tPandas: {pd.__version__}")
    print(f"\tScikit-learn: {sklearn.__version__}")
    print(f"\tMatplotlib: {sys.modules[plt.__package__].__version__}")
    print(f"\tSeaborn: {sns.__version__}")
    print(f"\tXGBoost: {xgb.__version__}")
    print(f"\tMLflow: {mlflow.__version__}")


def print_model(model_name, model, model_params):
    print(f" ")
    print(f"model_name: {model_name}")
    pprint(model)
    print(f"model_params:")
    pprint(model_params)


#%% Report functions -----------------------------------------------------------
# def make_pandas_profiling(df):
#     print(f"\nPandas profiling report start...")
#     config_file = pandas_profiling_dir / f"enriched_shipdb.yaml"
#     report_name = f"containershipdb_042021_enriched-profile.html"
#     pp_report = pp.ProfileReport(df_1, config_file=config_file)
#     pp_report.to_file(reports_dir / report_name)


def make_sweetviz_analyze(df):
    print(f"\nSweetViz analysis report start...")
    print(f"\tSweetViz: {sv.__version__}")
    print(f"Config SweetViz...")
    sv.config_parser.set(section="Layout", option="show_logo", value="0")
    feature_config = sv.FeatureConfig(skip=None, force_text=None)
    report_name = f"containershipdb_042021_enriched-analyze.html"
    report = sv.analyze(
        source=df,
        target_feat=None,
        feat_cfg=feature_config,
        pairwise_analysis="on",
    )
    report.show_html(reports_dir / report_name)


def make_sweetviz_compare(
    df_0,
    df_1,
    df_0_name="Raw data",
    df_1_name="After enriching",
    target_feat=None,
):
    print(f"\nSweetViz compare report start...")
    if df_0[target_feat].isna().sum() > 0:
        df_0[target_feat] = df_0[target_feat].fillna(-1)
    report_name = f"containershipdb_042021_enriched-compare.html"
    sv.config_parser.set(section="Layout", option="show_logo", value="0")
    feature_config = sv.FeatureConfig(skip=None, force_text=None)
    report = sv.compare(
        [df_0, df_0_name],
        [df_1, df_1_name],
        target_feat=target_feat,
        feat_cfg=feature_config,
    )
    report.show_html(reports_dir / report_name)
    return report


def make_dataprep_report(
    df,
    title,
    name,
):
    report_name = name
    report = create_report(df, title)
    report.save(filename=report_name, to=reports_dir)
    report.show_browser()


#%% Data functions -------------------------------------------------------------
def load_data(file_to_load):
    print(f"\nLoad data from <{file_to_load}> ...")
    df = pd.read_csv(
        data_processed_dir / file_to_load,
        header=0,
    )
    print(f"\n{file_to_load}: Loaded {len(df)} rows X {len(df.columns)} cols")
    # print(f"View random 4 rows:")
    # display(df.sample(4).T)
    return df


def get_data(df, target, features=None):
    print(f"\nGet data with the target [{target}]:")

    def print_out(df, X, y):
        print(f"\tOriginal X.shape: {df.shape}")
        print(f"\tOutput X.shape: {X.shape}, y.shape: {y.shape}")

    if target == "displacement":
        print(f"If [displacement] <= [deadweight] then cast [displacement] to NAN...")
        num_na_before = df["displacement"].isna().sum()
        df.loc[df["displacement"] <= df["deadweight"], "displacement"] = np.nan
        num_na_after = df["displacement"].isna().sum()
        print(f"\t#{num_na_after - num_na_before} displacement to NANs...")

    print(f"Split by target variable will been enriched...")
    df_dev = df[df[target].notna()]
    df_enrich = df[df[target].isna()]
    print(f"Was split accurate? {(df.shape[0]-df_dev.shape[0]==df_enrich.shape[0])}")
    # Pop the target & imo
    y_dev = df_dev.pop(target)
    y_enrich = df_enrich.pop(target)
    imo_dev = df_dev.pop("imo")
    imo_enrich = df_enrich.pop("imo")

    if features is None:
        print(f"Get data with all features...")
        X_dev = df_dev
        X_enrich = df_enrich
        print(f"Data for model training:")
        print_out(df_dev, X_dev, y_dev)
        print(f"Data for enriching w/ {target}=NaN:")
        print_out(df_enrich, X_enrich, y_enrich)
    else:
        print(f"Get data with {len(features)} features...")
        X_dev = df_dev.drop(df_dev.columns.difference(features), axis=1)
        X_enrich = df_enrich.drop(df_enrich.columns.difference(features), axis=1)
        print(f"Data for model training:")
        print_out(df_dev, X_dev, y_dev)
        print(f"Data for enriching w/ {target}=NaN:")
        print_out(df_enrich, X_enrich, y_enrich)
    X_dev = X_dev.fillna(-1)  # *
    X_enrich = X_enrich.fillna(-1)  # *
    return X_dev, y_dev, X_enrich, y_enrich, (imo_dev, imo_enrich)


def insert_predicted(y_pred, imo, df, file_to_save):
    print(f"Insert predicted values to original data...")
    target = y_pred.name
    imo_pred = pd.DataFrame(imo[1])
    idx = list(list(imo_pred.index))  # get indexes

    # Combain into one df
    df_pred = pd.concat(
        [imo_pred, pd.DataFrame(y_pred).set_index(imo_pred.index)],
        axis=1,
    )
    pred_to_save = f"{target}_predicted.csv"
    df_pred.to_csv(models_dir / "predictions" / pred_to_save, header=True, index=True)

    # View original data
    df.loc[idx, ["imo", target]]
    df.loc[idx, target] = list(y_pred)
    df.loc[idx, ["imo", target]]
    # Save enriched data
    df.to_csv(data_processed_dir / file_to_save, header=True, index=False)
    return df


#%% Model fitting functions ----------------------------------------------------
def fit_model(X_fit, y_fit, model_pack):
    model = model_pack[0]
    model_name = model_pack[1]
    model_params = model_pack[2]
    target = y_fit.name
    print(f"\nFit the {model_name} for {target}...")
    print(f"Model_params:")
    pprint(model_params)
    model.fit(X_fit, y_fit)
    return (model, target)


def train_model(X_fit, y_fit, model_pack, log_model=True, save_model_aside=True, set_name="train"):
    model = model_pack[0]
    model_name = model_pack[1]
    model_params = model_pack[2]
    target = y_fit.name
    input_example = X_fit.iloc[[0, -1]]  # first and last rows
    print(f"\nTrain final model {model_name} for [{target}]...")
    print(f"Train dataset: X={X_fit.shape}, y={y_fit.shape}")
    print(f"Model_params:")
    pprint(model_params)

    # * Setup MLflow tracking server
    exp_id = mlflow_set_exp_id("Model:Train")
    run_name = f"{model_name}-{target}"
    # * Enable autologging
    if model_name[:3] == "XGB":
        mlflow.xgboost.autolog(log_input_examples=True, log_models=log_model)
    else:
        mlflow.sklearn.autolog(log_input_examples=True, log_models=log_model)

    ##* Fit model with MLflow logging
    with mlflow.start_run(experiment_id=exp_id, run_name=run_name) as run:
        run_id = run.info.run_id
        model = model.fit(X_fit, y_fit)

        if model_name[:3] == "XGB":
            # * Log a XGBoost model
            if log_model:
                mlflow.xgboost.log_model(
                    xgb_model=model,
                    artifact_path="model",
                    input_example=input_example,
                )
            # * Disable autologging
            mlflow.xgboost.autolog(disable=True)
        else:
            mlflow.sklearn.autolog(disable=True)

        ##* Log metrics
        scores = {}
        scores_dev = calculate_metrics(model, X_fit, y_fit, set_name)
        scores.update(scores_dev)
        mlflow.log_metrics(scores)
        mlflow.log_params(model_params)

    if save_model_aside:
        # Save model separatly
        folder = save_mlmodel_aside(model, target, input_example, run_id)
        print(f"\nExperiment ID: {exp_id}")
        print(f"Run ID: {run_id}")
        print(f"Folder: {folder}")
    return model, (exp_id, run_id)


def predict_model(model, X, target):
    print(f"\nMake prediction and convert to Pandas Series...")
    y_pred = model.predict(X)
    y_pred = np.round(y_pred)
    y_pred = np.absolute(y_pred)
    y_pred = pd.Series(y_pred, name=target)
    return y_pred


#%% Plot functions -------------------------------------------------------------
def plot_density(y_true, y_pred, model, mlflow_ids):
    model_name = type(model).__name__
    target = y_true.name
    sns.set(style="darkgrid")
    kwargs = dict(hist=True, kde=True, bins=25)
    fig = plt.figure(figsize=(15, 10))
    plt.title(f"Predicted [{target}] density for {model_name}")
    plt.xlabel(f"Predicted {target}")
    sns.distplot(
        y_true,
        color="blue",
        label="Develop",
        kde_kws={"linewidth": 2},
    )
    sns.distplot(
        y_pred,
        color="red",
        label="Predicted",
        kde_kws={"linewidth": 3},
    )
    display(plt.legend())
    # Log density plot
    print(f"Log density plot to MLFlow Tracking...")
    exp_id = mlflow_ids[0]
    run_id = mlflow_ids[1]
    mlflow.start_run(experiment_id=exp_id, run_id=run_id)
    path = f"./plots/{model_name}-{target}_{run_id}.png"
    mlflow.log_figure(fig, path)
    mlflow.end_run()


#%% Features functions ---------------------------------------------------------
def get_feature_importance_xgb(model_fit, threshold=0.99):
    model = model_fit[0]
    target = model_fit[1]
    # Get the importance dictionary (by gain) from the booster
    cols_imp = model.get_booster().get_score(importance_type="gain")
    cols_imp = dict(sorted(cols_imp.items(), key=lambda item: item[1], reverse=True))
    # Get relative and cumulutive of relative importance
    relative_imp = [x / sum(cols_imp.values()) for x in cols_imp.values()]
    cumrel_imp = np.cumsum([x for x in relative_imp])
    # Create custom dict w/ col:relative importance: cumImportance
    importance = {}
    for i, (col, val) in enumerate(cols_imp.items()):
        # print(i, col, val)
        feature_key = f"{col}: {relative_imp[i]:.2%}: {cumrel_imp[i]:.1%}"
        feature_imp = round(val, 2)
        feature = {feature_key: feature_imp}
        importance.update(feature)

    # Get features limited by threshold
    if threshold is not None:
        threshold = threshold
        threshold_idx = np.argmin(cumrel_imp < threshold)
        imp_features = list(cols_imp.keys())[:threshold_idx]
        if not imp_features:
            print(f"\tThe <imp_features> list is EMPTY...")
            print(f"\tGet the 4th top features...")
            imp_features = list(cols_imp.keys())[:4]
    else:
        imp_features = list(cols_imp.keys())
    # Print out
    print(f"#Important features w/ threshold: {threshold} : {len(imp_features)}")
    print(f"Feature importance : Cumulutive Importance: Feature")
    dict_features = {}  # dict for saving info
    for i, feat in enumerate(imp_features):
        print(f"\t{relative_imp[i]:.2%} : {cumrel_imp[i]:.2%} : {feat}")
        dict_features.update(
            {
                feat: {
                    "relative_imp": relative_imp[i],
                    "cumulate_imp": cumrel_imp[i],
                    "gain": list(cols_imp.values())[i],
                }
            }
        )

    # Plot feature importance
    fig, ax = plt.subplots(figsize=(10, 20))
    first_line = f"Feature Importance for 'TEU' measured in Gain"
    second_line = f"Feature: %Relative importance: %Cumulutive importance"
    title = f"{first_line}\n{second_line}"
    importance_type = type
    xgb.plot_importance(
        importance,
        ax=ax,
        title=title,
        xlabel="Score",
        importance_type=importance_type,
        show_values=True,
    )
    # display(fig)
    fig.show()

    # Save feature list
    df_features = pd.DataFrame.from_dict(dict_features, orient="index")
    dir = models_dir / "params"
    file = f"features_{threshold}_{target}.csv"
    df_features.to_csv(dir / file, index_label="feature")
    print(f"List of feature importance saved in {dir/file}")

    return imp_features


#%% Auxilary functions ---------------------------------------------------------
def split_dev_data(X_dev, y_dev):
    X_train, X_val, y_train, y_val = model_selection.train_test_split(
        X_dev,
        y_dev,
        train_size=0.9,
        random_state=rnd_state,
        shuffle=True,
    )
    #%% Print out
    print(f"X_train: {X_train.shape[1]} variables & {X_train.shape[0]} records.")
    print(f"X_val: {X_val.shape[1]} variables & {X_val.shape[0]} records.")
    print(f"X_dev: {X_dev.shape[1]} variables & {X_dev.shape[0]} records.")
    return X_train, X_val, y_train, y_val


def calculate_metrics(model, X, y, set_name):
    print(f"\nCalculate model metrics...")
    model_name = type(model).__name__
    target = y.name
    y_pred = model.predict(X)
    # Calculate metrics
    max = metrics.max_error(y_true=y, y_pred=y_pred)
    mae = metrics.mean_absolute_error(y_true=y, y_pred=y_pred)
    medae = metrics.median_absolute_error(y_true=y, y_pred=y_pred)
    mape = my_mape(y_true=y, y_pred=y_pred)
    rmse = metrics.mean_squared_error(y_true=y, y_pred=y_pred, squared=False)
    explvar = metrics.explained_variance_score(y_true=y, y_pred=y_pred)
    print(f"On {set_name} set the {model_name} for {target} gives following metrics:")
    print(f"\tMax Absolute Error:: {max:.4f}")
    print(f"\tMean Absolute Error:: {mae:.4f}")
    print(f"\tMedian Absolute Error:: {medae:.4f}")
    print(f"\tMean Absolute Percentage Error:: {mape:.4f}")
    print(f"\tRoot Mean Squared Error:: {rmse:.4f}")
    print(f"\tExplained Variance(R^2):: {explvar:.4f}")
    eval_scores = {
        f"max_{set_name}": max,
        f"mae_{set_name}": mae,
        f"medae_{set_name}": medae,
        f"mape_{set_name}": mape,
        f"rmse_{set_name}": rmse,
        f"r2_{set_name}": explvar,
    }
    return eval_scores


def log_model_metrics(model, X_train, y_train, X_val, y_val):
    print(f"Calculate metrics and log them...")
    # model_name = type(model).__name__
    scores = {}
    scores_train = calculate_metrics(model, X_train, y_train, "train")
    scores_val = calculate_metrics(model, X_val, y_val, "val")
    scores.update(scores_train)
    scores.update(scores_val)
    mlflow.log_metrics(scores)
    return True


def log_model_residuals(model, X_train, y_train, X_val, y_val):
    tic = time.time()
    print(f"\nCalculate and log model's residuals...")
    fig = plot_residuals_errors(model, X_train, y_train, X_val, y_val)
    mlflow.log_figure(fig, "./plots/residuals_errors.png")
    min, sec = divmod(time.time() - tic, 60)
    print(f"Calculating residuals took: {int(min)}min {int(sec)}sec")
    display(fig)
    return True


#%% ------------------- Bayesian Optimization Module----------------------------
def bayesian_search_cv(
    X_fit,
    y_fit,
    model_pack,
    bayes_space,
    n_bayes_params_sets=50,
    n_splits=5,
    log_model=False,
    log_metrics_residuals=True,
):
    print(f"\n---------- Bayesian optimization of hyper-params started....")
    print(f"Parameters' space:")
    pprint(bayes_space)
    model = model_pack[0]
    model_name = model_pack[1]
    target = y_fit.name

    # Define metrics
    # scoring = metrics.make_scorer(score_func=my_mape, greater_is_better=True)
    scoring = metrics.get_scorer("neg_root_mean_squared_error")
    # Define num. of CV splits and K-repeats
    cv = model_selection.RepeatedKFold(
        n_splits=n_splits,
        n_repeats=1,  #  n times with different randomization in each repetition.
        random_state=rnd_state,
    )
    # Define bayesian space search
    bayes_search = BayesSearchCV(
        model,
        search_spaces=bayes_space,
        n_iter=n_bayes_params_sets,
        scoring=scoring,
        cv=cv,
        refit=True,
        return_train_score=True,
        n_jobs=-1,
        verbose=3,
        random_state=rnd_state,
    )
    # Callback handler
    def on_step(optim_result):
        """Print scores after each iteration while performing optimization"""
        score = bayes_search.best_score_
        print(f"\nCurrent best score: {score} ...")

    # * Setup MLflow tracking server
    exp_id = mlflow_set_exp_id("Model:Fit")
    run_name = f"{model_name}-{target}: bayes"
    # mlflow_del_default_experiment() #! Don't delete. Needs for MLproject
    # * Enable autologging
    if model_name[:3] == "XGB":
        mlflow.xgboost.autolog(log_models=log_model)
    else:
        mlflow.sklearn.autolog(log_models=log_model)
    # * Fit model with MLflow logging
    with mlflow.start_run(experiment_id=exp_id, run_name=run_name) as run:
        run_id = run.info.run_id
        tic = time.time()
        model_bayes_search = bayes_search.fit(
            X_fit,
            y_fit,
            callback=on_step,
        )
        min, sec = divmod(time.time() - tic, 60)
        # * Disable autologging
        if model_name[:3] == "XGB":
            mlflow.xgboost.autolog(disable=True)
        else:
            mlflow.sklearn.autolog(disable=True)

        if log_metrics_residuals:
            X_train, X_val, y_train, y_val = split_dev_data(X_fit, y_fit)
            # Log metrics
            print(f"Bayesian search took: {int(min)}min {int(sec)}sec")
            print(f"\nLog metrics...")
            mlflow.log_metric("best_CV_score", bayes_search.best_score_)
            log_model_metrics(model_bayes_search, X_train, y_train, X_val, y_val)
            # Log params
            model_best_params = model_bayes_search.best_params_
            search_params = {"n_iter": n_bayes_params_sets, "n_splits": n_splits}
            model_best_params.update(search_params)
            mlflow.log_params(model_best_params)
            # Log residuals
            log_model_residuals(model_bayes_search, X_train, y_train, X_val, y_val)

    print(f"\nBayesian search: Best score is:\n\t {model_bayes_search.best_score_}")
    print(f"Bayesian search: Best params are:\n\t")
    pprint(model_bayes_search.best_params_)

    model_pack = (
        model_bayes_search.best_estimator_,
        type(model_bayes_search.best_estimator_).__name__,
        dict(model_bayes_search.best_params_),
    )
    winsound.Beep(frequency=2000, duration=300)
    return model_pack, (exp_id, run_id)


#%% Saving functuons -----------------------------------------------------------
def save_params(model_pack, id, params_dir, target):
    model_name = model_pack[1]
    bs_bp = model_pack[2]
    dir = params_dir / "params"

    if id[1] is not None:
        t = mlflow_get_run_data(id[1])[4]
        time_point = f"{t[0]}{t[1]:02d}{t[2]:02d}-{t[3]:02d}{t[4]:02d}"
    else:
        time_point = time.strftime("%Y%m%d-%H%M")

    file = f"{model_name}-{target}_best-params.json"
    print(f"\nSave parameters found...")
    if bs_bp is not None:
        with open(dir / file, "w+") as f:
            json.dump(bs_bp, f)


def save_mlmodel_aside(model, target, input_example, run_id):
    print(f"\nSave aside a trained model at MLflow's format...")
    model_name = type(model).__name__
    if run_id is not None:
        t = mlflow_get_run_data(run_id)[4]
        # time_point = f"{t[0]}{t[1]:02d}{t[2]:02d}-{t[3]:02d}{t[4]:02d}"
        folder = f"{model_name}-{target}_{run_id}"
    else:
        # time_point = time.strftime("%Y%m%d-%H%M")
        model_name = type(model).__name__
        folder = f"{model_name}-{target}"

    path = models_dir / "mlmodels" / folder
    mlflow.sklearn.save_model(
        model,
        path,
        conda_env=None,
        mlflow_model=None,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        input_example=input_example,
    )
    print(f"MLflow model saved aside.You can find it in: {path} folder...")
    return folder


#%% Main =======================================================================
def main(
    file_to_load,
    target,
    file_to_save,
    n_bayes_params_sets=50,
    n_splits=5,
    log_metrics_residuals=True,
):
    """Performs FITTING MODELS on DEVELOPMENT set with searching an
    optimal hyper-parameters with Baeysian search.

    Args:
        file_to_load ([type]): name of csv file w/ processed development data
        target ([type]): target to predict
        * n_bayes_params_sets (int, optional): Number of parameter settings that are sampled.\
            Defaults to 20.
        * n_splits (int, optional): Num. of splits in Cross-Validation strategy.\
            Defaults to 5.
        * log_metrics_residuals (bool, optional): Toogle to calculate and mlflow log model's
        metricd and rediduals.\
            Defaults to True.
    """

    print(f"---------- START: Get feature importance --------------------------")
    print_versions()
    set_pd_options()
    set_matlotlib_params()
    # * Data
    df = load_data(file_to_load)
    df_get_glimpse(df)

    X_dev, y_dev, _, _, _ = get_data(df, target, features=None)
    # * Feature importance
    model_pack_xgb = get_model_xgb()
    model_fi = fit_model(X_dev, y_dev, model_pack_xgb)
    imp_features = get_feature_importance_xgb(model_fi, threshold=0.995)
    # * Bayesian optimization
    print(f"---------- Bayesian optimization: Hyper-params searching ----------")
    model_pack = get_model_xgb()
    _, _, bayes_space = get_param_search_xgb()
    bayes_space.pop("booster")  # use for XGB
    bayes_space.pop("gamma")  # use for XGB

    X_dev, y_dev, X_enrich, _, imo = get_data(df, target, features=imp_features)
    model_pack_opted, id = bayesian_search_cv(
        X_dev,
        y_dev,
        model_pack,
        bayes_space,
        n_bayes_params_sets=n_bayes_params_sets,
        n_splits=n_splits,
        log_metrics_residuals=log_metrics_residuals,
    )
    save_params(model_pack_opted, id, models_dir, target)
    # * Train model
    print(f"---------- Train model with hyper-params found --------------------")
    model_train, mlflow_ids = train_model(X_dev, y_dev, model_pack_opted)
    y_pred = predict_model(model_train, X_enrich, target)
    plot_density(y_dev, y_pred, model_train, mlflow_ids)
    df_enriched = insert_predicted(y_pred, imo, df, file_to_save)
    reset_pd_options()
    print(f"\n!!! DONE: Fit model !!!")
    winsound.Beep(frequency=3000, duration=300)

    return df_enriched


#%% RUN ========================================================================
if __name__ == "__main__":  #! Make sure that Tracking Server has been run.
    """Variables line will should be predicted for "Draft Survey Method":
    [X] deadweight: #3, %0.1
    [X] draught: #40, %0.7
    [X] teu: #43, %0.8
    [X] ballast_water: #2594, %47.9
    [X] freshwater: #4289, %79.2
    [X] displacement: #5041, %93.1
    """
    warnings.filterwarnings("ignore")
    rnd_state = 42
    vars = [
        "containers",  # staring data set derived from raw data
        "gross_tonnage",  # resides <shipdb_api_features>
        "deadweight",  #! resides <shipdb_api_features>
        "year_of_built",  # resides <shipdb_api_features>
        "length_overall",  # resides <shipdb_api_features>
        "breadth_moulded",
        "draught",  #! resides <shipdb_api_features>
        "teu",  #! resides <shipdb_api_features>
        "depth",
        "depth_moulded",  # resides <shipdb_api_features>
        "max_draught",  # 10
        "engine_power",  # resides <shipdb_api_features>
        "net_tonnage",  # resides <shipdb_api_features>
        "breadth_registered",  # resides <shipdb_api_features>
        "eng_total_kw",  # resides <shipdb_api_features>
        "eng_total_bhp",  # resides <shipdb_api_features>
        "ballast",
        "ballast_water",  #!
        "fuel",
        "fuel_oil",
        "freshwater",  #! 20
        "diesel_oil",
        "lube_oil",
        "displacement",  #!
        "gas_oil",
        # * other variables
        "max_speed",
        "main_engine_rpm",
        "main_engine_kw",
        "freeboard",
        "light_displacement_tonnes",
        "decks_number",  # 30
        "length_bp",
        "draught_tropical",
        "draught_winter",
        "ais_dim_d",
        "ais_dim_a",
        "ais_dim_c",
        "ais_dim_b",
        "freeboard_winter",
        "freeboard_tropical",
        "service_speed",  # 40
        "freeboard_summer",
        "deck_teu",
        "hold_teu",
        "reefer_pts",
        "length_registered",
        "reefer_teu",
        "bulkheads",
        "main_engine_power",
        "cargo_holds",
        "beam_extreme",  # 50
        "breadth_extreme",
        "trial_speed",
    ]
    for f, t in zip(enumerate(vars[:-1]), vars[1:]):
        """Set the 'vars' index to start predict the next variable
        i.e. to start from begining - set 'f[0] < 0'
            to start from "length_bp" - set 'f[0] < 30'
        """
        if f[0] < 0:
            pass
        else:
            file_to_load = f"shipdb_042021_{f[0]}_{f[1]}.csv"
            file_to_save = f"shipdb_042021_{f[0]+1}_{t}.csv"
            target = f"{t}"
            print(f"\nDATA: {file_to_load} \ TARGET: {t} \ SAVE IN: {file_to_save}")

            df = main(
                file_to_load=file_to_load,
                target=target,
                file_to_save=file_to_save,
                n_bayes_params_sets=50,  #!
                n_splits=5,  #!
            )
    # Make reports
    df_0 = load_data(file_to_load="shipdb_042021_0_containers.csv")
    df_0 = df_0[vars[1:]]
    df_1 = df[vars[1:]]

    make_sweetviz_analyze(df)
    make_sweetviz_compare(df_0, df)
    make_dataprep_report(
        df,
        title="ContainerShipDB 042021 after enriching",
        name="containershipdb_042021_enriched-dataprep",
    )

#%% Auxilary cell
# file_to_load = "shipdb_042021_22_lube_oil.csv"
# target = "displacement"
# file_to_save = "shipdb_042021_23_displacement_separated_pred.csv"

#%%
