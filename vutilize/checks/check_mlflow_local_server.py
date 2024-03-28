"""
    * NO NEED TO CHANGE *
    Helps to run MLflow Tracking Server with Registry DB
    into defined, particular directory and check how it works.
    To check, firstly it is necessary to run the server.

    @author: mikhail.galkin
"""

""" mlflow server [OPTIONS]
    mlflow server --backend-store-uri <PATH> --default-artifact-root <URI>

    [OPTIONS]:
    --backend-store-uri <PATH>
        URI to which to persist experiment and run data.
        Acceptable URIs are SQLAlchemy-compatible database connection strings
        (e.g. ‘sqlite:///path/to/file.db’)
        or local filesystem URIs (e.g. ‘file:///absolute/path/to/directory’).
        By default, data will be logged to the ./mlruns directory.

    --default-artifact-root <URI>
        Local or S3 URI to store artifacts, for new experiments.
        Note that this flag does not impact already-created experiments.
        Default: Within file store, if a file:/ URI is provided.
        If a sql backend is used, then this option is required.

To ensure that have a consistent artifact_location
if you are running experiments on your localhost, it is recommended decide
ahead of time where you want artifact store and the backend store to be.
Once decided, launch tracking server with those arguments. For example:
> mlflow server --backend-store-uri sqlite:///path/to/file.db \
    --default-artifact-root file://absolute_path_to_directory/mlruns
"""
#%% Load libraries
import joblib
import mlflow
from mlflow.tracking import MlflowClient

if __name__ == "__main__":
    with open("test.file", "wb") as f:
        test = range(20)
        joblib.dump(test, f)

    EXPERIMENT = "Default"

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    exp_info = MlflowClient().get_experiment_by_name(EXPERIMENT)

    if exp_info:
        exp_id = exp_info.experiment_id
    else:
        exp_id = MlflowClient().create_experiment(EXPERIMENT)

    with mlflow.start_run(experiment_id=exp_id, run_name="Test") as run:
        run_id = run.info.run_id

        params = {"n_estimators": 3, "random_state": 0}
        mlflow.log_params(params)
        mlflow.log_artifact("test.file")

        print(f"artifact_uri={mlflow.get_artifact_uri()}")
        print(f"run_id={run_id}")

#%%
